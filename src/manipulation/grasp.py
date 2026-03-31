import dataclasses as dc
import time
import typing

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt_connect import (
    RRTConnectConfig,
    postprocess_rrt_path,
    rrt_connect_plan,
)


@dc.dataclass(frozen=True)
class GraspVariant:
    """
    Relative offset from pregrasp to grasp target.
    """
    approach_depth_m: float = 0.0
    lateral_offset_m: float = 0.0
    vertical_offset_m: float = 0.0
    yaw_offset_rad: float = 0.0


@dc.dataclass(frozen=True)
class GraspOptions:
    """
    Planning options for the grasp primitive.
    """
    position_tol: float = 0.005
    theta_tol: float = 0.05
    ik_soft_starts: int = 16
    ik_soft_start_sigma: float = 0.08
    ik_soft_start_seed: int = 0
    # Prefer a Cartesian straight-line insertion from pregrasp to grasp.
    approach_prefer_straight_line: bool = True
    approach_linear_step_m: float = 0.01
    approach_linear_max_waypoints: int = 20
    # World-frame retreat offsets applied to pregrasp for grasp->retreat.
    retreat_offset_world_y_m: float = 0.0
    retreat_offset_world_z_m: float = 0.0
    variants: tuple[GraspVariant, ...] = (
        # Keep default variants centered on the brick to avoid side-offset misses.
        GraspVariant(approach_depth_m=0.10, lateral_offset_m=0.00, yaw_offset_rad=0.00),
        GraspVariant(approach_depth_m=0.095, lateral_offset_m=0.00, yaw_offset_rad=0.00),
        GraspVariant(approach_depth_m=0.105, lateral_offset_m=0.00, yaw_offset_rad=0.00),
    )


@dc.dataclass
class GraspPrimitivePlan:
    """
    Planned motion for pregrasp -> grasp -> post-grasp retreat.
    """
    selected_variant_index: int
    selected_variant: GraspVariant
    X_WG_grasp: RigidTransform
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray
    q_postgrasp_retreat: np.ndarray
    path_pregrasp_to_grasp: list[np.ndarray]
    path_grasp_to_pregrasp: list[np.ndarray]


@dc.dataclass
class GraspPrimitiveResult:
    success: bool
    plan: GraspPrimitivePlan | None
    failure_reasons: list[str]


def _make_grasp_pose(X_WG_pregrasp: RigidTransform, variant: GraspVariant) -> RigidTransform:
    R_WG_pre = X_WG_pregrasp.rotation()
    p_WG_pre = X_WG_pregrasp.translation()

    x_axis_W = R_WG_pre.matrix()[:, 0]
    y_axis_W = R_WG_pre.matrix()[:, 1]
    z_axis_W = R_WG_pre.matrix()[:, 2]

    p_WG_grasp = (
        p_WG_pre
        + float(variant.approach_depth_m) * y_axis_W
        + float(variant.lateral_offset_m) * x_axis_W
        + float(variant.vertical_offset_m) * z_axis_W
    )

    if abs(float(variant.yaw_offset_rad)) <= 1e-12:
        R_WG_grasp = R_WG_pre
    else:
        R_WYaw = RollPitchYaw(0.0, 0.0, float(variant.yaw_offset_rad)).ToRotationMatrix()
        R_WG_grasp = RotationMatrix(R_WYaw.matrix() @ R_WG_pre.matrix())

    return RigidTransform(R_WG_grasp, p_WG_grasp)


def _plan_rrt_segment(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    is_free: typing.Callable[[np.ndarray], bool],
    joints_lower_limits: np.ndarray,
    joints_upper_limits: np.ndarray,
    planner_config: RRTConnectConfig,
    planning_deadline_s: float | None = None,
    search_is_free: typing.Callable[[np.ndarray], bool] | None = None,
) -> list[np.ndarray]:
    raw_path = rrt_connect_plan(
        q_start=q_start,
        q_goal=q_goal,
        is_free=is_free,
        search_is_free=search_is_free,
        joints_lower_limits=joints_lower_limits,
        joints_upper_limits=joints_upper_limits,
        **planner_config.to_plan_kwargs(deadline_s=planning_deadline_s),
    )
    return postprocess_rrt_path(
        raw_path,
        is_free,
        planner_config=planner_config,
        deadline_s=planning_deadline_s,
    )


def _make_retreat_pose(X_WG_pregrasp: RigidTransform, options: GraspOptions) -> RigidTransform:
    p_WG_pre = X_WG_pregrasp.translation()
    p_WG_retreat = p_WG_pre + np.array(
        [0.0, float(options.retreat_offset_world_y_m), float(options.retreat_offset_world_z_m)]
    )
    return RigidTransform(X_WG_pregrasp.rotation(), p_WG_retreat)


def _plan_cartesian_approach_segment(
    plant,
    root_context_current,
    iiwa_instance,
    wsg_instance,
    is_free: typing.Callable[[np.ndarray], bool],
    q_pregrasp: np.ndarray,
    X_WG_pregrasp: RigidTransform,
    X_WG_grasp: RigidTransform,
    options: GraspOptions,
    variant_index: int,
    planning_deadline_s: float | None = None,
) -> list[np.ndarray]:
    q_pregrasp = np.asarray(q_pregrasp, dtype=float).reshape(7)
    p_start = np.asarray(X_WG_pregrasp.translation(), dtype=float).reshape(3)
    p_goal = np.asarray(X_WG_grasp.translation(), dtype=float).reshape(3)
    distance = float(np.linalg.norm(p_goal - p_start))
    step_m = max(1e-4, float(options.approach_linear_step_m))
    n_segments = max(1, int(np.ceil(distance / step_m)))
    n_segments = min(n_segments, max(1, int(options.approach_linear_max_waypoints)))

    path = [q_pregrasp.copy()]
    q_seed = q_pregrasp.copy()
    # Keep orientation fixed at the grasp orientation and interpolate only translation.
    R_approach = X_WG_grasp.rotation()

    for segment_idx in range(1, n_segments + 1):
        alpha = float(segment_idx) / float(n_segments)
        p_waypoint = (1.0 - alpha) * p_start + alpha * p_goal
        X_waypoint = RigidTransform(R_approach, p_waypoint)
        q_waypoint = solve_iiwa_ik_for_gripper_pose(
            plant=plant,
            root_context_current=root_context_current,
            iiwa_instance=iiwa_instance,
            wsg_instance=wsg_instance,
            desired_end_effector=X_waypoint,
            q_iiwa_seed=q_seed,
            position_tol=options.position_tol,
            theta_tol=options.theta_tol,
            max_soft_starts=options.ik_soft_starts,
            soft_start_sigma=options.ik_soft_start_sigma,
            soft_start_random_seed=(
                options.ik_soft_start_seed + 10_000 * variant_index + segment_idx
            ),
            deadline_s=planning_deadline_s,
        )
        if not is_free(q_waypoint):
            raise RuntimeError(
                f"straight-line approach collision at waypoint {segment_idx}/{n_segments}"
            )
        q_waypoint = np.asarray(q_waypoint, dtype=float).reshape(7)
        path.append(q_waypoint.copy())
        q_seed = q_waypoint

    return path


def plan_grasp_primitive(
    plant,
    root_context_current,
    iiwa_instance,
    wsg_instance,
    is_free: typing.Callable[[np.ndarray], bool],
    joints_lower_limits: np.ndarray,
    joints_upper_limits: np.ndarray,
    q_pregrasp: np.ndarray,
    X_WG_pregrasp: RigidTransform,
    planner_config: RRTConnectConfig,
    options: GraspOptions | None = None,
    is_free_retreat: typing.Callable[[np.ndarray], bool] | None = None,
    prepare_retreat_checker: typing.Callable[[np.ndarray], None] | None = None,
    planning_deadline_s: float | None = None,
) -> GraspPrimitiveResult:
    """
    Plans the grasp primitive only:
      pregrasp -> grasp -> post-grasp retreat

    The approach leg (pregrasp -> grasp) always uses `is_free`.
    The retreat leg (grasp -> pregrasp) may use `is_free_retreat` when provided.
    If provided, `prepare_retreat_checker` is called with q_grasp before planning
    the retreat leg.
    """
    options = options or GraspOptions()
    is_free_retreat = is_free_retreat or is_free
    q_pregrasp = np.asarray(q_pregrasp, dtype=float).reshape(7)
    joints_lower_limits = np.asarray(joints_lower_limits, dtype=float).reshape(7)
    joints_upper_limits = np.asarray(joints_upper_limits, dtype=float).reshape(7)

    failure_reasons: list[str] = []

    for variant_index, variant in enumerate(options.variants):
        if planning_deadline_s is not None and time.perf_counter() > float(planning_deadline_s):
            failure_reasons.append("grasp primitive planning timeout exceeded")
            break
        try:
            X_WG_grasp = _make_grasp_pose(X_WG_pregrasp, variant)

            q_grasp = solve_iiwa_ik_for_gripper_pose(
                plant=plant,
                root_context_current=root_context_current,
                iiwa_instance=iiwa_instance,
                wsg_instance=wsg_instance,
                desired_end_effector=X_WG_grasp,
                q_iiwa_seed=q_pregrasp,
                position_tol=options.position_tol,
                theta_tol=options.theta_tol,
                max_soft_starts=options.ik_soft_starts,
                soft_start_sigma=options.ik_soft_start_sigma,
                soft_start_random_seed=options.ik_soft_start_seed + variant_index,
                deadline_s=planning_deadline_s,
            )

            if not is_free(q_grasp):
                failure_reasons.append(f"variant_{variant_index}: IK solution not collision-free")
                continue

            path_pregrasp_to_grasp = None
            if options.approach_prefer_straight_line:
                try:
                    path_pregrasp_to_grasp = _plan_cartesian_approach_segment(
                        plant=plant,
                        root_context_current=root_context_current,
                        iiwa_instance=iiwa_instance,
                        wsg_instance=wsg_instance,
                        is_free=is_free,
                        q_pregrasp=q_pregrasp,
                        X_WG_pregrasp=X_WG_pregrasp,
                        X_WG_grasp=X_WG_grasp,
                        options=options,
                        variant_index=variant_index,
                        planning_deadline_s=planning_deadline_s,
                    )
                except Exception as straight_exc:
                    try:
                        path_pregrasp_to_grasp = _plan_rrt_segment(
                            q_start=q_pregrasp,
                            q_goal=q_grasp,
                            is_free=is_free,
                            joints_lower_limits=joints_lower_limits,
                            joints_upper_limits=joints_upper_limits,
                            planner_config=planner_config,
                            planning_deadline_s=planning_deadline_s,
                        )
                    except Exception as rrt_exc:
                        raise RuntimeError(
                            f"straight-line approach failed ({straight_exc}); "
                            f"RRT fallback failed ({rrt_exc})"
                        )
            else:
                path_pregrasp_to_grasp = _plan_rrt_segment(
                    q_start=q_pregrasp,
                    q_goal=q_grasp,
                    is_free=is_free,
                    joints_lower_limits=joints_lower_limits,
                    joints_upper_limits=joints_upper_limits,
                    planner_config=planner_config,
                    planning_deadline_s=planning_deadline_s,
                )

            if prepare_retreat_checker is not None:
                prepare_retreat_checker(np.asarray(q_grasp, dtype=float).reshape(7))

            X_WG_retreat = _make_retreat_pose(X_WG_pregrasp, options)
            q_retreat = solve_iiwa_ik_for_gripper_pose(
                plant=plant,
                root_context_current=root_context_current,
                iiwa_instance=iiwa_instance,
                wsg_instance=wsg_instance,
                desired_end_effector=X_WG_retreat,
                q_iiwa_seed=q_grasp,
                position_tol=options.position_tol,
                theta_tol=options.theta_tol,
                max_soft_starts=options.ik_soft_starts,
                soft_start_sigma=options.ik_soft_start_sigma,
                soft_start_random_seed=options.ik_soft_start_seed + 1000 + variant_index,
                deadline_s=planning_deadline_s,
            )
            if not is_free_retreat(q_retreat):
                failure_reasons.append(
                    f"variant_{variant_index}: post-grasp retreat IK solution not collision-free"
                )
                continue

            path_grasp_to_pregrasp = _plan_rrt_segment(
                q_start=q_grasp,
                q_goal=q_retreat,
                is_free=is_free_retreat,
                joints_lower_limits=joints_lower_limits,
                joints_upper_limits=joints_upper_limits,
                planner_config=planner_config,
                planning_deadline_s=planning_deadline_s,
            )

            return GraspPrimitiveResult(
                success=True,
                plan=GraspPrimitivePlan(
                    selected_variant_index=variant_index,
                    selected_variant=variant,
                    X_WG_grasp=X_WG_grasp,
                    q_pregrasp=q_pregrasp.copy(),
                    q_grasp=np.asarray(q_grasp, dtype=float).copy(),
                    q_postgrasp_retreat=np.asarray(q_retreat, dtype=float).copy(),
                    path_pregrasp_to_grasp=[np.asarray(q, dtype=float).copy() for q in path_pregrasp_to_grasp],
                    path_grasp_to_pregrasp=[np.asarray(q, dtype=float).copy() for q in path_grasp_to_pregrasp],
                ),
                failure_reasons=failure_reasons,
            )
        except Exception as exc:
            failure_reasons.append(f"variant_{variant_index}: {exc}")

    return GraspPrimitiveResult(success=False, plan=None, failure_reasons=failure_reasons)
