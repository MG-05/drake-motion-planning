"""Benchmarks for the planners in this repo.

Two experiments, both run on the real Drake shelf scene from
configs/scenes/starter_env.yaml:

1. `single_query`: home -> pregrasp joint-space planning (goal inside the
   shelf mouth) with three planners:
     - vanilla RRT                    (src/planning/rrt.py)
     - RRT-Connect, fixed step       (adaptive controller disabled)
     - RRT-Connect, adaptive step    (default AdaptiveStepConfig)

2. `pipeline`: the full pick-and-place ManipulationFSM (plan only), with and
   without the deterministic clutter-interpolation stages (straight-line
   grasp approach, carry escape, pre-drop insertion). Without them, the RRT
   has to solve extraction, transport, and insertion in one search.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/benchmark_planners.py run
    PYTHONPATH=. .venv/bin/python scripts/benchmark_planners.py plot

Results are stored in assets/benchmarks/results.json and figures are written
to assets/.
"""

import argparse
import dataclasses as dc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pydrake.common.yaml import yaml_load_typed
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import ModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import AddMultibodyPlant
from pydrake.systems.framework import DiagramBuilder

from src.main import Scenario, default_scenario_path, _compute_iiwa_joint_limits, _compute_wsg_planning_configs
from src.manipulation.grasp import GraspOptions, GraspVariant
from src.manipulation.manipulation_fsm import ManipulationFSM, ManipulationOptions
from src.manipulation.pregrasp import compute_pregrasp_pose_for_brick, get_floating_body
from src.planning.collision import is_collision_free
from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt import rrt_plan
from src.planning.rrt_connect import (
    AdaptiveStepConfig,
    RRTConnectConfig,
    postprocess_rrt_path,
    rrt_connect_plan,
)

SHELF_LEVEL_TO_BRICK_POSITION = {
    "low": np.array([0.0, 0.15, 0.030]),
    "mid": np.array([0.0, 0.15, 0.30]),
    "high": np.array([0.0, 0.15, 0.56]),
}
DROP_POSITION_W = np.array([0.0, 0.05, 0.78])

RESULTS_PATH = REPO_ROOT / "assets" / "benchmarks" / "results.json"


class PlannerTimeout(Exception):
    pass


def _with_deadline(is_free_fn, deadline_s):
    """Wrap a state checker so any planner respects a wall-clock cutoff."""

    def wrapped(q):
        if time.perf_counter() > deadline_s:
            raise PlannerTimeout()
        return bool(is_free_fn(q))

    return wrapped


def _path_length(path) -> float:
    path = [np.asarray(q, dtype=float) for q in path]
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


@dc.dataclass
class Scene:
    plant: object
    scene_graph: object
    root_context: object
    plant_context: object
    iiwa: object
    wsg: object
    brick_instance: object
    q_start: np.ndarray
    joints_lower: np.ndarray
    joints_upper: np.ndarray
    checkers: dict
    X_WG_pregrasp: RigidTransform
    X_WG_drop: RigidTransform


def build_scene(shelf_level: str) -> Scene:
    scenario = yaml_load_typed(
        schema=Scenario,
        filename=str(default_scenario_path()),
        child_name="StarterEnv",
        defaults=Scenario(),
    )
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(
        plant_config=scenario.plant_config,
        scene_graph_config=scenario.scene_graph_config,
        builder=builder,
    )
    ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives), plant=plant
    )
    plant.Finalize()
    diagram = builder.Build()
    root_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)

    brick_instance = plant.GetModelInstanceByName("foam_brick")
    brick_body = get_floating_body(plant, brick_instance)
    X_WB = RigidTransform(
        RollPitchYaw(0.0, 0.0, math.radians(-90.0)).ToRotationMatrix(),
        SHELF_LEVEL_TO_BRICK_POSITION[shelf_level],
    )
    plant.SetFreeBodyPose(plant_context, brick_body, X_WB)

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")
    q_start = plant.GetPositions(plant_context, iiwa).copy()
    q_wsg_open, q_wsg_closed = _compute_wsg_planning_configs(
        plant.GetPositions(plant_context, wsg).copy()
    )
    joints_lower, joints_upper = _compute_iiwa_joint_limits(plant, iiwa)

    # The same four collision-checker semantics as src/main.py.
    checkers = {
        "strict": is_collision_free(
            plant=plant, scene_graph=scene_graph, root_context=root_context,
            iiwa_instance=iiwa, wsg_instance=wsg, q_wsg_instance=q_wsg_closed,
            min_clearance=0.01, pair_range=0.05,
        ),
        "grasp": is_collision_free(
            plant=plant, scene_graph=scene_graph, root_context=root_context,
            iiwa_instance=iiwa, wsg_instance=wsg, q_wsg_instance=q_wsg_open,
            min_clearance=0.0, pair_range=0.05,
            ignore_model_instances=[brick_instance],
        ),
        "deapproach": is_collision_free(
            plant=plant, scene_graph=scene_graph, root_context=root_context,
            iiwa_instance=iiwa, wsg_instance=wsg, q_wsg_instance=q_wsg_closed,
            min_clearance=0.0, pair_range=0.12,
            extra_checked_model_instances=[brick_instance],
        ),
        "carry": is_collision_free(
            plant=plant, scene_graph=scene_graph, root_context=root_context,
            iiwa_instance=iiwa, wsg_instance=wsg, q_wsg_instance=q_wsg_closed,
            min_clearance=0.017, pair_range=0.07,
            extra_checked_model_instances=[brick_instance],
        ),
        "drop_preplace_clearance": is_collision_free(
            plant=plant, scene_graph=scene_graph, root_context=root_context,
            iiwa_instance=iiwa, wsg_instance=wsg, q_wsg_instance=q_wsg_closed,
            min_clearance=0.0, pair_range=0.30,
            extra_checked_model_instances=[brick_instance],
        ),
    }

    X_WG_pregrasp = compute_pregrasp_pose_for_brick(
        plant=plant,
        plant_context=plant_context,
        iiwa_instance=iiwa,
        brick_body=brick_body,
        fingertip_clearance_m=0.04,
        wsg_body_to_fingertips_m=0.14,
    )
    X_WG_drop = RigidTransform(X_WG_pregrasp.rotation(), DROP_POSITION_W)

    return Scene(
        plant=plant, scene_graph=scene_graph, root_context=root_context,
        plant_context=plant_context, iiwa=iiwa, wsg=wsg,
        brick_instance=brick_instance, q_start=q_start,
        joints_lower=joints_lower, joints_upper=joints_upper,
        checkers=checkers, X_WG_pregrasp=X_WG_pregrasp, X_WG_drop=X_WG_drop,
        )


def solve_pregrasp_ik(scene: Scene) -> np.ndarray:
    return np.asarray(
        solve_iiwa_ik_for_gripper_pose(
            plant=scene.plant,
            root_context_current=scene.root_context,
            iiwa_instance=scene.iiwa,
            wsg_instance=scene.wsg,
            desired_end_effector=scene.X_WG_pregrasp,
            q_iiwa_seed=scene.q_start,
            position_tol=0.002,
            theta_tol=0.035,
            max_soft_starts=20,
            soft_start_sigma=0.08,
            soft_start_random_seed=0,
        ),
        dtype=float,
    ).reshape(7)


# ---------------------------------------------------------------------------
# Experiment 1: single-query planner comparison (home -> pregrasp)
# ---------------------------------------------------------------------------

def run_single_query(levels, seeds, timeout_s):
    rrt_connect_base = RRTConnectConfig()
    fixed_adaptive = dc.replace(rrt_connect_base.adaptive_step_config, enabled=False)
    results = []
    for level in levels:
        scene = build_scene(level)
        q_goal = solve_pregrasp_ik(scene)
        is_free = scene.checkers["strict"]
        for planner in ("rrt", "rrt_connect_fixed", "rrt_connect_adaptive"):
            for seed in seeds:
                deadline = time.perf_counter() + timeout_s
                guarded = _with_deadline(is_free, deadline)
                t0 = time.perf_counter()
                entry = {
                    "level": level, "planner": planner, "seed": int(seed),
                    "timeout_s": timeout_s,
                }
                try:
                    if planner == "rrt":
                        path = rrt_plan(
                            scene.q_start, q_goal, guarded,
                            scene.joints_lower, scene.joints_upper,
                            step_size=0.1, goal_sample_rate=0.1,
                            max_iters=200_000, edge_resolution=0.02,
                            goal_tolerance=0.1,
                        )
                    else:
                        adaptive = (
                            rrt_connect_base.adaptive_step_config
                            if planner == "rrt_connect_adaptive"
                            else fixed_adaptive
                        )
                        config = dc.replace(
                            rrt_connect_base,
                            random_seed=int(seed),
                            adaptive_step_config=adaptive,
                        )
                        raw = rrt_connect_plan(
                            q_start=scene.q_start,
                            q_goal=q_goal,
                            is_free=is_free,
                            search_is_free=guarded,
                            joints_lower_limits=scene.joints_lower,
                            joints_upper_limits=scene.joints_upper,
                            **config.to_plan_kwargs(deadline_s=deadline),
                        )
                        path = postprocess_rrt_path(
                            raw, is_free, planner_config=config, deadline_s=deadline
                        )
                    entry.update(
                        success=True,
                        time_s=time.perf_counter() - t0,
                        path_length_rad=_path_length(path),
                        waypoints=len(path),
                    )
                except (PlannerTimeout, TimeoutError):
                    entry.update(success=False, time_s=timeout_s, reason="timeout")
                except Exception as exc:  # planner reported failure
                    entry.update(
                        success=False,
                        time_s=time.perf_counter() - t0,
                        reason=f"{type(exc).__name__}",
                    )
                results.append(entry)
                print(
                    f"[single_query] {level:<4} {planner:<22} seed={seed} "
                    f"success={entry['success']} t={entry['time_s']:.2f}s",
                    flush=True,
                )
    return results


# ---------------------------------------------------------------------------
# Experiment 2: full pipeline with/without deterministic clutter stages
# ---------------------------------------------------------------------------

def _make_fsm(scene: Scene, *, deterministic_stages: bool, seed: int, max_time_s: float) -> ManipulationFSM:
    grasp_options = GraspOptions(
        ik_soft_starts=20,
        ik_soft_start_sigma=0.08,
        ik_soft_start_seed=0,
        approach_prefer_straight_line=deterministic_stages,
        approach_linear_step_m=0.01,
        approach_linear_max_waypoints=20,
        retreat_offset_world_y_m=0.0,
        retreat_offset_world_z_m=0.02,
        # Same variant set as src/main.py: centered grasps first, then
        # above-centerline fallbacks for low-ceiling compartments.
        variants=(
            GraspVariant(approach_depth_m=0.10),
            GraspVariant(approach_depth_m=0.095),
            GraspVariant(approach_depth_m=0.105),
            GraspVariant(
                approach_depth_m=0.10,
                vertical_offset_m=0.02,
                retreat_extra_lift_m=0.02,
            ),
            GraspVariant(
                approach_depth_m=0.105,
                vertical_offset_m=0.02,
                retreat_extra_lift_m=0.02,
            ),
        ),
    )
    options = ManipulationOptions(
        ik_soft_starts=20,
        local_ik_soft_starts=2,
        ik_soft_start_sigma=0.08,
        ik_soft_start_seed=0,
        max_planning_time_s=max_time_s,
        home_to_pregrasp_time_budget_s=0.15 * max_time_s,
        grasp_primitive_time_budget_s=0.20 * max_time_s,
        pregrasp_to_drop_time_budget_s=0.60 * max_time_s,
        rrt=RRTConnectConfig(
            final_validation_edge_resolution=0.01,
            random_seed=int(seed),
        ),
        drop_candidate_time_budget_s=min(60.0, 0.5 * max_time_s),
        drop_preplace_clearance_threshold_m=0.20,
        max_drop_candidates=20,
        enable_drop_transport_bridges=False,
        enable_carry_escape=deterministic_stages,
        enable_drop_preplace=deterministic_stages,
        grasp_options=grasp_options,
    )
    return ManipulationFSM(
        plant=scene.plant,
        root_context_current=scene.root_context,
        iiwa_instance=scene.iiwa,
        wsg_instance=scene.wsg,
        is_free=scene.checkers["strict"],
        joints_lower_limits=scene.joints_lower,
        joints_upper_limits=scene.joints_upper,
        is_free_grasp=scene.checkers["grasp"],
        is_free_deapproach=scene.checkers["deapproach"],
        is_free_carry=scene.checkers["carry"],
        drop_preplace_clearance_source=scene.checkers["drop_preplace_clearance"],
        q_wsg_carry=_compute_wsg_planning_configs(
            scene.plant.GetPositions(scene.plant_context, scene.wsg).copy()
        )[1],
        carry_payload_instance=scene.brick_instance,
        options=options,
    )


def run_pipeline(levels, seeds, timeout_s):
    results = []
    for level in levels:
        for deterministic_stages in (True, False):
            # Fresh scene per configuration so checker payload state is clean.
            scene = build_scene(level)
            for seed in seeds:
                fsm = _make_fsm(
                    scene,
                    deterministic_stages=deterministic_stages,
                    seed=seed,
                    max_time_s=timeout_s,
                )
                t0 = time.perf_counter()
                fsm_result = fsm.run(
                    q_home=scene.q_start,
                    X_WG_pregrasp=scene.X_WG_pregrasp,
                    X_WG_drop=scene.X_WG_drop,
                )
                entry = {
                    "level": level,
                    "deterministic_stages": deterministic_stages,
                    "seed": int(seed),
                    "timeout_s": timeout_s,
                    "success": bool(fsm_result.success),
                    "time_s": time.perf_counter() - t0,
                    "timings_s": {k: float(v) for k, v in fsm_result.timings_s.items()},
                }
                if not fsm_result.success:
                    entry["error"] = (fsm_result.error_message or "")[:300]
                results.append(entry)
                print(
                    f"[pipeline] {level:<4} deterministic={deterministic_stages} "
                    f"seed={seed} success={entry['success']} t={entry['time_s']:.1f}s",
                    flush=True,
                )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Figure styling: Okabe-Ito colorblind-safe palette, applied semantically.
# Warm colors (vermillion/amber) mark baselines and failure-prone settings;
# cool blues mark the sampling-based planners of this repo; greens mark the
# deterministic stages; gray marks neutral infrastructure (IK).
COLOR_BASELINE = "#D55E00"       # vermillion: caution / baseline
COLOR_INTERMEDIATE = "#E69F00"   # amber: intermediate baseline
COLOR_OURS = "#0072B2"           # deep blue: trust / this repo's planner
COLOR_SEARCH_LIGHT = "#56B4E9"   # sky blue: lighter search stage
COLOR_DETERMINISTIC = "#009E73"  # green: deterministic / safe
COLOR_DETERMINISTIC_ALT = "#7FBFA0"
COLOR_DETERMINISTIC_ALT2 = "#B7DDC9"
COLOR_NEUTRAL = "#8C8C8C"        # gray: neutral infrastructure
TEXT_COLOR = "#333333"

PLANNER_LABELS = {
    "rrt": "RRT (vanilla)",
    "rrt_connect_fixed": "RRT-Connect (fixed step)",
    "rrt_connect_adaptive": "RRT-Connect (adaptive)",
}
PLANNER_COLORS = {
    "rrt": COLOR_BASELINE,
    "rrt_connect_fixed": COLOR_INTERMEDIATE,
    "rrt_connect_adaptive": COLOR_OURS,
}
STAGE_KEYS = [
    ("plan_home_to_pregrasp", "home → pregrasp RRT", COLOR_SEARCH_LIGHT),
    ("plan_grasp_primitive", "grasp primitive (deterministic)", COLOR_DETERMINISTIC),
    ("plan_carry_escape", "carry escape (deterministic)", COLOR_DETERMINISTIC_ALT),
    ("plan_drop_anchor_ik", "drop anchor IK", COLOR_NEUTRAL),
    ("plan_drop_preplace", "pre-drop insertion (deterministic)", COLOR_DETERMINISTIC_ALT2),
    ("plan_drop_rrt", "carry transport RRT", COLOR_OURS),
]


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _median(values):
    return float(np.median(np.asarray(values, dtype=float))) if values else float("nan")


def plot_single_query(results, out_path: Path, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=200)
    planners = list(PLANNER_LABELS)
    width = 0.26
    x = np.arange(len(levels))
    timeout_s = results[0]["timeout_s"] if results else 30.0

    for j, planner in enumerate(planners):
        heights, labels, hatches = [], [], []
        for level in levels:
            trials = [
                r for r in results if r["planner"] == planner and r["level"] == level
            ]
            successes = [r for r in trials if r["success"]]
            n = len(trials)
            if successes:
                heights.append(_median([r["time_s"] for r in successes]))
                hatches.append(None)
            else:
                heights.append(timeout_s)
                hatches.append("//")
            labels.append(f"{len(successes)}/{n}")
        bars = ax.bar(
            x + (j - 1) * width,
            heights,
            width,
            label=PLANNER_LABELS[planner],
            color=PLANNER_COLORS[planner],
            edgecolor="white",
        )
        for bar, label, hatch in zip(bars, labels, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_alpha(0.45)
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{level} shelf" for level in levels])
    ax.set_ylabel("median planning time (s, log scale)")
    ax.set_title(
        "home → pregrasp planning time by planner\n"
        f"(labels show successes / trials; hatched = all trials hit {timeout_s:.0f}s cutoff)"
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_COLOR)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def plot_pipeline(results, out_path: Path, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=200)
    x = np.arange(len(levels))
    width = 0.34
    timeout_s = results[0]["timeout_s"] if results else 90.0
    configs = [
        (True, "with deterministic clutter stages", COLOR_OURS),
        (False, "RRT-only (stages disabled)", COLOR_BASELINE),
    ]
    for j, (deterministic, label, color) in enumerate(configs):
        heights, labels, hatches = [], [], []
        for level in levels:
            trials = [
                r
                for r in results
                if r["deterministic_stages"] == deterministic and r["level"] == level
            ]
            successes = [r for r in trials if r["success"]]
            if successes:
                heights.append(_median([r["time_s"] for r in successes]))
                hatches.append(None)
            else:
                heights.append(timeout_s)
                hatches.append("//")
            labels.append(f"{len(successes)}/{len(trials)}")
        bars = ax.bar(
            x + (j - 0.5) * width, heights, width,
            label=label, color=color, edgecolor="white",
        )
        for bar, lbl, hatch in zip(bars, labels, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_alpha(0.45)
            ax.annotate(
                lbl,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TEXT_COLOR,
            )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{level} shelf" for level in levels])
    ax.set_ylabel("median total planning time (s, log scale)")
    ax.set_title(
        "end-to-end pick-and-place planning time\n"
        f"(labels show successes / trials; hatched = all trials failed within {timeout_s:.0f}s)"
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_COLOR)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def plot_stage_breakdown(results, out_path: Path, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=200)
    y = np.arange(len(levels))
    left = np.zeros(len(levels))
    for key, label, color in STAGE_KEYS:
        medians = []
        for level in levels:
            trials = [
                r
                for r in results
                if r["deterministic_stages"] and r["level"] == level and r["success"]
            ]
            medians.append(
                _median([r["timings_s"].get(key, 0.0) for r in trials]) if trials else 0.0
            )
        medians = np.nan_to_num(np.asarray(medians))
        ax.barh(y, medians, left=left, label=label, color=color, edgecolor="white")
        left += medians

    ax.set_yticks(y)
    ax.set_yticklabels([f"{level} shelf" for level in levels])
    ax.invert_yaxis()
    ax.set_xlabel("median wall time (s)")
    ax.set_title("where planning time goes (full pipeline, successful runs)")
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        labelcolor=TEXT_COLOR,
    )
    _style_axes(ax)
    ax.grid(axis="y", color="none")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "plot"])
    parser.add_argument("--levels", nargs="+", default=["low", "mid", "high"])
    parser.add_argument("--single_query_seeds", type=int, default=5)
    parser.add_argument("--pipeline_seeds", type=int, default=3)
    parser.add_argument("--single_query_timeout_s", type=float, default=30.0)
    parser.add_argument("--pipeline_timeout_s", type=float, default=90.0)
    parser.add_argument("--skip_single_query", action="store_true")
    parser.add_argument("--skip_pipeline", action="store_true")
    args = parser.parse_args()

    if args.command == "run":
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results = {"levels": args.levels}
        if RESULTS_PATH.exists():
            results.update(json.loads(RESULTS_PATH.read_text()))
            results["levels"] = args.levels
        if not args.skip_single_query:
            results["single_query"] = run_single_query(
                args.levels,
                range(args.single_query_seeds),
                args.single_query_timeout_s,
            )
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
        if not args.skip_pipeline:
            results["pipeline"] = run_pipeline(
                args.levels, range(args.pipeline_seeds), args.pipeline_timeout_s
            )
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print(f"wrote {RESULTS_PATH}")
    else:
        results = json.loads(RESULTS_PATH.read_text())
        levels = results.get("levels", ["low", "mid", "high"])
        assets = REPO_ROOT / "assets"
        if results.get("single_query"):
            plot_single_query(
                results["single_query"], assets / "benchmark_planners.png", levels
            )
        if results.get("pipeline"):
            # Only plot shelf levels where the task itself is feasible (at least
            # one configuration succeeded); infeasible levels are reported in
            # the results JSON instead.
            pipeline_levels = [
                level
                for level in levels
                if any(
                    r["success"] for r in results["pipeline"] if r["level"] == level
                )
            ]
            plot_pipeline(
                results["pipeline"],
                assets / "benchmark_pipeline_ablation.png",
                pipeline_levels,
            )
            plot_stage_breakdown(
                results["pipeline"],
                assets / "benchmark_stage_breakdown.png",
                pipeline_levels,
            )


if __name__ == "__main__":
    main()
