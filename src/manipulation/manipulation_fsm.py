import dataclasses as dc
import enum
import time
import typing

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from src.manipulation.grasp import GraspOptions, GraspPrimitivePlan, plan_grasp_primitive
from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt_connect import (
    RRTConnectConfig,
    edge_is_free,
    postprocess_rrt_path,
    rrt_connect_plan,
)


class ManipulationState(str, enum.Enum):
    HOME_READY = "home_ready"
    PLAN_HOME_TO_PREGRASP = "plan_home_to_pregrasp"
    RUN_GRASP_PRIMITIVE = "run_grasp_primitive"
    PLAN_PREGRASP_TO_DROP = "plan_pregrasp_to_drop"
    RELEASE_AT_DROP = "release_at_drop"
    DONE = "done"
    FAILED = "failed"


@dc.dataclass(frozen=True)
class ManipulationOptions:
    """
    Task-level options for pick-and-place planning.
    """
    position_tol: float = 0.002
    theta_tol: float = 0.035
    ik_soft_starts: int = 20
    local_ik_soft_starts: int = 4
    ik_soft_start_sigma: float = 0.08
    ik_soft_start_seed: int = 0
    rrt: RRTConnectConfig = dc.field(default_factory=RRTConnectConfig)
    max_planning_time_s: float | None = 60.0
    home_to_pregrasp_time_budget_s: float | None = None
    grasp_primitive_time_budget_s: float | None = None
    pregrasp_to_drop_time_budget_s: float | None = None
    drop_candidate_time_budget_s: float | None = 60.0
    max_drop_candidates: int | None = 20
    drop_rrt_candidate_batch_sizes: tuple[int, ...] = (1, 5, 20)
    drop_rrt_seed_offsets: tuple[int, ...] = (2, 3, 4)
    enable_carry_escape: bool = True
    carry_escape_clearance_threshold_m: float | None = None
    carry_escape_try_home_staging: bool = True
    axial_pullback_step_m: float = 0.015
    carry_escape_max_pullback_m: float = 0.20
    carry_escape_open_clearance_margin_m: float | None = 0.02
    enable_drop_preplace: bool = True
    drop_preplace_clearance_threshold_m: float | None = None
    drop_preplace_max_pullback_m: float = 0.20
    drop_preplace_open_clearance_margin_m: float | None = 0.02
    enable_drop_transport_bridges: bool = False
    drop_transport_bridge_backoff_m: tuple[float, ...] = (0.25, 0.17, 0.09)
    drop_transport_bridge_low_z_offset_m: tuple[float, ...] = (-0.13, -0.03)
    drop_transport_bridge_high_z_offset_m: tuple[float, ...] = (0.07, 0.17)
    drop_xy_offsets_m: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.03, 0.0),
        (-0.03, 0.0),
        (0.0, 0.03),
        (0.0, -0.03),
    )
    drop_z_offsets_m: tuple[float, ...] = (0.0, 0.03, 0.06, 0.10)
    drop_yaw_offsets_rad: tuple[float, ...] = (0.0, 0.035, -0.035)
    grasp_options: GraspOptions = dc.field(default_factory=GraspOptions)


@dc.dataclass
class ManipulationPlan:
    q_home: np.ndarray
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray
    q_drop: np.ndarray
    drop_candidate_index: int
    X_WG_drop_selected: RigidTransform
    path_home_to_pregrasp: list[np.ndarray]
    grasp_plan: GraspPrimitivePlan
    path_pregrasp_to_drop: list[np.ndarray]
    gripper_events: list[str]


@dc.dataclass
class ManipulationResult:
    success: bool
    final_state: ManipulationState
    plan: ManipulationPlan | None
    error_message: str | None
    transition_history: list[ManipulationState]
    timings_s: dict[str, float] = dc.field(default_factory=dict)


@dc.dataclass
class AxialClearancePathResult:
    path: list[np.ndarray]
    final_clearance_m: float | None
    stop_reason: str


@dc.dataclass
class DropTransportSharedBridges:
    bridge2_configs: list[tuple[float, np.ndarray, bool]]
    bridge1_configs: dict[float, list[np.ndarray]]


def build_axial_clearance_path(
    q_anchor: np.ndarray,
    *,
    step_m: float,
    max_pullback_m: float,
    solve_step: typing.Callable[[float, np.ndarray, int], np.ndarray],
    state_is_free: typing.Callable[[np.ndarray], bool],
    edge_is_free_fn: typing.Callable[[np.ndarray, np.ndarray], bool],
    estimate_clearance_fn: typing.Callable[[np.ndarray], float] | None = None,
    target_clearance_m: float | None = None,
    stop_predicate: typing.Callable[[np.ndarray], bool] | None = None,
    check_deadline: typing.Callable[[], None] | None = None,
) -> AxialClearancePathResult:
    q_anchor = np.asarray(q_anchor, dtype=float).copy()
    if float(step_m) <= 0.0:
        raise ValueError("step_m must be positive.")
    if float(max_pullback_m) < 0.0:
        raise ValueError("max_pullback_m must be nonnegative.")

    def _maybe_check_deadline() -> None:
        if check_deadline is not None:
            check_deadline()

    def _maybe_estimate_clearance(q: np.ndarray) -> float | None:
        if estimate_clearance_fn is None:
            return None
        return float(estimate_clearance_fn(q))

    path = [q_anchor.copy()]
    final_clearance_m = _maybe_estimate_clearance(q_anchor)

    _maybe_check_deadline()
    if stop_predicate is not None and stop_predicate(q_anchor):
        return AxialClearancePathResult(
            path=path,
            final_clearance_m=final_clearance_m,
            stop_reason="predicate",
        )
    if (
        target_clearance_m is not None
        and final_clearance_m is not None
        and final_clearance_m >= float(target_clearance_m)
    ):
        return AxialClearancePathResult(
            path=path,
            final_clearance_m=final_clearance_m,
            stop_reason="clearance",
        )

    pullback_distances = list(
        np.arange(float(step_m), float(max_pullback_m) + 1e-12, float(step_m))
    )
    if not pullback_distances or pullback_distances[-1] < float(max_pullback_m) - 1e-12:
        pullback_distances.append(float(max_pullback_m))

    q_prev = q_anchor.copy()
    for step_index, pullback_m in enumerate(pullback_distances, start=1):
        _maybe_check_deadline()
        try:
            q_next = solve_step(float(pullback_m), q_prev.copy(), step_index)
        except TimeoutError:
            raise
        except Exception:
            continue

        q_next = np.asarray(q_next, dtype=float).copy()
        _maybe_check_deadline()
        if not state_is_free(q_next):
            continue
        _maybe_check_deadline()
        if not edge_is_free_fn(q_prev, q_next):
            continue

        path.append(q_next.copy())
        q_prev = q_next
        final_clearance_m = _maybe_estimate_clearance(q_prev)

        _maybe_check_deadline()
        if stop_predicate is not None and stop_predicate(q_prev):
            return AxialClearancePathResult(
                path=path,
                final_clearance_m=final_clearance_m,
                stop_reason="predicate",
            )
        if (
            target_clearance_m is not None
            and final_clearance_m is not None
            and final_clearance_m >= float(target_clearance_m)
        ):
            return AxialClearancePathResult(
                path=path,
                final_clearance_m=final_clearance_m,
                stop_reason="clearance",
            )

    return AxialClearancePathResult(
        path=path,
        final_clearance_m=final_clearance_m,
        stop_reason="exhausted",
    )


class ManipulationFSM:
    """
    Orchestrates:
      home -> pregrasp -> grasp -> pregrasp -> drop
    """

    def __init__(
        self,
        plant,
        root_context_current,
        iiwa_instance,
        wsg_instance,
        is_free: typing.Callable[[np.ndarray], bool],
        joints_lower_limits: np.ndarray,
        joints_upper_limits: np.ndarray,
        is_free_grasp: typing.Callable[[np.ndarray], bool] | None = None,
        is_free_deapproach: typing.Callable[[np.ndarray], bool] | None = None,
        is_free_carry: typing.Callable[[np.ndarray], bool] | None = None,
        drop_preplace_clearance_source: typing.Callable[[np.ndarray], bool] | None = None,
        q_wsg_carry: np.ndarray | None = None,
        carry_payload_instance=None,
        carry_payload_carrier_frame_name: str = "body",
        options: ManipulationOptions | None = None,
    ):
        self.plant = plant
        self.root_context_current = root_context_current
        self.iiwa_instance = iiwa_instance
        self.wsg_instance = wsg_instance
        self.is_free = is_free
        self.is_free_grasp = is_free_grasp or is_free
        self.is_free_carry = is_free_carry or self.is_free_grasp
        self.is_free_deapproach = is_free_deapproach or self.is_free_carry
        self.drop_preplace_clearance_source = (
            drop_preplace_clearance_source or self.is_free_carry
        )
        self.q_wsg_carry = None if q_wsg_carry is None else np.asarray(q_wsg_carry, dtype=float).copy()
        self.carry_payload_instance = carry_payload_instance
        self.carry_payload_carrier_frame_name = str(carry_payload_carrier_frame_name)
        self.joints_lower_limits = np.asarray(joints_lower_limits, dtype=float).reshape(7)
        self.joints_upper_limits = np.asarray(joints_upper_limits, dtype=float).reshape(7)
        self.options = options or ManipulationOptions()
        self._history: list[ManipulationState] = []

    def _transition(self, state: ManipulationState) -> None:
        self._history.append(state)

    def _configure_carry_payload_attachment(self, q_grasp: np.ndarray) -> None:
        if self.carry_payload_instance is None:
            return

        configure_fns: list[typing.Callable[..., None]] = []
        seen = set()
        for checker in (
            self.is_free_deapproach,
            self.is_free_carry,
            self.drop_preplace_clearance_source,
        ):
            if checker is None:
                continue
            checker_id = id(checker)
            if checker_id in seen:
                continue
            seen.add(checker_id)
            configure_fn = getattr(checker, "configure_attached_model", None)
            if callable(configure_fn):
                configure_fns.append(configure_fn)
        if not configure_fns:
            return

        ctx = self.root_context_current.Clone()
        plant_context = self.plant.GetMyMutableContextFromRoot(ctx)

        q_grasp = np.asarray(q_grasp, dtype=float).reshape(7)
        self.plant.SetPositions(plant_context, self.iiwa_instance, q_grasp)
        if self.q_wsg_carry is not None:
            self.plant.SetPositions(plant_context, self.wsg_instance, self.q_wsg_carry)

        payload_body = None
        for body_index in self.plant.GetBodyIndices(self.carry_payload_instance):
            body = self.plant.get_body(body_index)
            if body.is_floating():
                payload_body = body
                break
        if payload_body is None:
            raise RuntimeError(
                f"Carry payload model instance {self.carry_payload_instance} has no floating body"
            )

        carrier_frame = self.plant.GetFrameByName(
            self.carry_payload_carrier_frame_name,
            self.wsg_instance,
        )
        X_WC = self.plant.CalcRelativeTransform(
            plant_context, self.plant.world_frame(), carrier_frame
        )
        X_WB = self.plant.CalcRelativeTransform(
            plant_context, self.plant.world_frame(), payload_body.body_frame()
        )
        X_CB = X_WC.inverse() @ X_WB
        for configure_fn in configure_fns:
            configure_fn(
                model_instance=self.carry_payload_instance,
                carrier_frame=carrier_frame,
                X_CB=X_CB,
            )

    def _solve_anchor_pose_ik(
        self,
        X_WG_target: RigidTransform,
        q_seed: np.ndarray,
        planning_deadline_s: float | None = None,
        *,
        soft_start_seed_offset: int = 0,
    ) -> np.ndarray:
        return solve_iiwa_ik_for_gripper_pose(
            plant=self.plant,
            root_context_current=self.root_context_current,
            iiwa_instance=self.iiwa_instance,
            wsg_instance=self.wsg_instance,
            desired_end_effector=X_WG_target,
            q_iiwa_seed=q_seed,
            position_tol=self.options.position_tol,
            theta_tol=self.options.theta_tol,
            max_soft_starts=self.options.ik_soft_starts,
            soft_start_sigma=self.options.ik_soft_start_sigma,
            soft_start_random_seed=self.options.ik_soft_start_seed + int(soft_start_seed_offset),
            deadline_s=planning_deadline_s,
        )

    @staticmethod
    def _format_pose_translation(X_WG_target: RigidTransform) -> str:
        p = np.asarray(X_WG_target.translation(), dtype=float).reshape(3)
        return "[" + ", ".join(f"{coord:.3f}" for coord in p) + "]"

    def _require_pose_reachable(
        self,
        *,
        pose_label: str,
        X_WG_target: RigidTransform,
        q_seed: np.ndarray,
        planning_deadline_s: float | None = None,
        soft_start_seed_offset: int = 0,
    ) -> np.ndarray:
        try:
            return self._solve_anchor_pose_ik(
                X_WG_target=X_WG_target,
                q_seed=q_seed,
                planning_deadline_s=planning_deadline_s,
                soft_start_seed_offset=soft_start_seed_offset,
            )
        except TimeoutError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"{pose_label} location is not in reach "
                f"(target position {self._format_pose_translation(X_WG_target)})."
            ) from exc

    def _plan_rrt(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        is_free_fn: typing.Callable[[np.ndarray], bool] | None = None,
        search_is_free_fn: typing.Callable[[np.ndarray], bool] | None = None,
        planning_deadline_s: float | None = None,
        random_seed: int | None = None,
        return_goal_index: bool = False,
        debug_label: str | None = None,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], int]:
        is_free_fn = is_free_fn or self.is_free
        plan_kwargs = self.options.rrt.to_plan_kwargs(deadline_s=planning_deadline_s)
        if random_seed is not None:
            plan_kwargs["random_seed"] = int(random_seed)
        try:
            result = rrt_connect_plan(
                q_start=q_start,
                q_goal=q_goal,
                is_free=is_free_fn,
                search_is_free=search_is_free_fn,
                joints_lower_limits=self.joints_lower_limits,
                joints_upper_limits=self.joints_upper_limits,
                **plan_kwargs,
                return_goal_index=return_goal_index,
            )
        except Exception as exc:
            if debug_label is None:
                raise
            raise type(exc)(f"{debug_label}: {exc}") from exc
        if return_goal_index:
            raw_path, goal_index = result
            return (
                postprocess_rrt_path(
                    raw_path,
                    is_free_fn,
                    planner_config=self.options.rrt,
                    deadline_s=planning_deadline_s,
                ),
                goal_index,
            )
        return postprocess_rrt_path(
            result,
            is_free_fn,
            planner_config=self.options.rrt,
            deadline_s=planning_deadline_s,
        )

    @staticmethod
    def _resolve_stage_deadline(
        planning_deadline_s: float | None,
        *,
        stage_start_s: float,
        stage_budget_s: float | None,
    ) -> float | None:
        stage_deadline_s = planning_deadline_s
        if stage_budget_s is not None:
            local_deadline_s = stage_start_s + max(0.0, float(stage_budget_s))
            if stage_deadline_s is None:
                stage_deadline_s = local_deadline_s
            else:
                stage_deadline_s = min(stage_deadline_s, local_deadline_s)
        return stage_deadline_s

    @staticmethod
    def _check_stage_deadline(
        stage_name: str,
        *,
        planning_deadline_s: float | None,
        stage_deadline_s: float | None,
        stage_budget_s: float | None,
    ) -> None:
        if stage_deadline_s is None or time.perf_counter() <= float(stage_deadline_s):
            return
        if (
            stage_budget_s is not None
            and (
                planning_deadline_s is None
                or float(stage_deadline_s) < float(planning_deadline_s) - 1e-12
            )
        ):
            raise TimeoutError(
                f"{stage_name} exceeded stage budget of {float(stage_budget_s):.1f}s"
            )
        raise TimeoutError("Planning exceeded the overall planning deadline")

    @staticmethod
    def _raise_if_deadline_exceeded(
        deadline_s: float | None,
        *,
        context: str,
    ) -> None:
        if deadline_s is None or time.perf_counter() <= float(deadline_s):
            return
        raise TimeoutError(f"{context} exceeded its planning deadline")

    def _rrt_random_seed(self, offset: int = 0) -> int | None:
        base_seed = self.options.rrt.random_seed
        if base_seed is None:
            return None
        return int(base_seed) + int(offset)

    def _ordered_drop_rrt_seed_offsets(self, batch_size: int) -> tuple[int, ...]:
        seed_offsets = tuple(
            int(seed_offset)
            for seed_offset in self.options.drop_rrt_seed_offsets
        ) or (0,)
        preferred_seed_offset = 3 if int(batch_size) <= 5 else 2
        if preferred_seed_offset not in seed_offsets:
            return seed_offsets
        return (preferred_seed_offset,) + tuple(
            seed_offset
            for seed_offset in seed_offsets
            if seed_offset != preferred_seed_offset
        )

    def _iter_drop_candidates(self, X_WG_drop: RigidTransform):
        p_WG_nominal = X_WG_drop.translation()
        R_WG_nominal = X_WG_drop.rotation()

        def _is_zero(value: float) -> bool:
            return abs(float(value)) <= 1e-12

        nominal_xy_offsets = [
            (float(dx), float(dy))
            for dx, dy in self.options.drop_xy_offsets_m
            if _is_zero(dx) and _is_zero(dy)
        ]
        non_nominal_xy_offsets = [
            (float(dx), float(dy))
            for dx, dy in self.options.drop_xy_offsets_m
            if not (_is_zero(dx) and _is_zero(dy))
        ]
        nominal_z_offsets = [
            float(dz)
            for dz in self.options.drop_z_offsets_m
            if _is_zero(dz)
        ]
        non_nominal_z_offsets = [
            float(dz)
            for dz in self.options.drop_z_offsets_m
            if not _is_zero(dz)
        ]
        nominal_yaw_offsets = [
            float(yaw)
            for yaw in self.options.drop_yaw_offsets_rad
            if _is_zero(yaw)
        ]
        non_nominal_yaw_offsets = [
            float(yaw)
            for yaw in self.options.drop_yaw_offsets_rad
            if not _is_zero(yaw)
        ]

        ordered_translation_offsets: list[tuple[float, float, float]] = []
        seen_translation_offsets: set[tuple[float, float, float]] = set()

        def _append_translation_offsets(
            xy_offsets: list[tuple[float, float]],
            z_offsets: list[float],
        ) -> None:
            for dx, dy in xy_offsets:
                for dz in z_offsets:
                    offset = (float(dx), float(dy), float(dz))
                    if offset in seen_translation_offsets:
                        continue
                    seen_translation_offsets.add(offset)
                    ordered_translation_offsets.append(offset)

        _append_translation_offsets(nominal_xy_offsets, nominal_z_offsets)
        _append_translation_offsets(nominal_xy_offsets, non_nominal_z_offsets)
        _append_translation_offsets(non_nominal_xy_offsets, nominal_z_offsets)
        _append_translation_offsets(non_nominal_xy_offsets, non_nominal_z_offsets)

        ordered_yaw_offsets = nominal_yaw_offsets + non_nominal_yaw_offsets
        candidate_index = 0
        for yaw in ordered_yaw_offsets:
            if abs(float(yaw)) <= 1e-12:
                R_WG_candidate = R_WG_nominal
            else:
                R_WYaw = RollPitchYaw(0.0, 0.0, float(yaw)).ToRotationMatrix()
                R_WG_candidate = RotationMatrix(R_WYaw.matrix() @ R_WG_nominal.matrix())
            for dx, dy, dz in ordered_translation_offsets:
                p_WG_candidate = p_WG_nominal + np.array([float(dx), float(dy), float(dz)])
                yield candidate_index, RigidTransform(R_WG_candidate, p_WG_candidate)
                candidate_index += 1

    def _strict_edge_resolution(self) -> float:
        resolution = self.options.rrt.final_validation_edge_resolution
        if resolution is None:
            resolution = self.options.rrt.edge_resolution
        return max(1e-6, float(resolution))

    def _strict_edge_is_free(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        is_free_fn: typing.Callable[[np.ndarray], bool],
        planning_deadline_s: float | None = None,
    ) -> bool:
        return edge_is_free(
            is_free_fn,
            np.asarray(q_start, dtype=float).reshape(7),
            np.asarray(q_goal, dtype=float).reshape(7),
            resolution=self._strict_edge_resolution(),
            deadline_s=planning_deadline_s,
        )

    @staticmethod
    def _resolve_open_clearance_target(
        is_free_fn: typing.Callable[[np.ndarray], bool],
        absolute_threshold_m: float | None,
        open_margin_m: float | None,
    ) -> float | None:
        estimate_clearance = getattr(is_free_fn, "estimate_clearance", None)
        if not callable(estimate_clearance):
            return None
        if absolute_threshold_m is not None:
            return float(absolute_threshold_m)
        if open_margin_m is None:
            return None
        return float(getattr(is_free_fn, "minimum_clearance", 0.0)) + max(
            0.0, float(open_margin_m)
        )

    def _calc_carrier_frame_pose(
        self,
        q_iiwa: np.ndarray,
        q_wsg_instance: np.ndarray | None = None,
    ) -> RigidTransform:
        q_iiwa = np.asarray(q_iiwa, dtype=float).reshape(7)
        fk_root_context = self.root_context_current.Clone()
        plant_context = self.plant.GetMyMutableContextFromRoot(fk_root_context)
        self.plant.SetPositions(plant_context, self.iiwa_instance, q_iiwa)

        if self.wsg_instance is not None:
            if q_wsg_instance is None:
                q_wsg_instance = self.plant.GetPositions(
                    plant_context, self.wsg_instance
                ).copy()
            q_wsg_instance = np.asarray(q_wsg_instance, dtype=float).reshape(-1)
            self.plant.SetPositions(plant_context, self.wsg_instance, q_wsg_instance)

        carrier_frame = self.plant.GetFrameByName(
            self.carry_payload_carrier_frame_name,
            self.wsg_instance,
        )
        return self.plant.CalcRelativeTransform(
            plant_context,
            self.plant.world_frame(),
            carrier_frame,
        )

    def _plan_axial_clearance_path(
        self,
        q_anchor: np.ndarray,
        is_free_fn: typing.Callable[[np.ndarray], bool],
        *,
        clearance_source_fn: typing.Callable[[np.ndarray], bool] | None = None,
        absolute_clearance_threshold_m: float | None,
        open_clearance_margin_m: float | None,
        max_pullback_m: float,
        step_m: float,
        planning_deadline_s: float | None = None,
        stop_predicate: typing.Callable[[np.ndarray], bool] | None = None,
        soft_start_seed_base: int = 0,
    ) -> AxialClearancePathResult:
        q_anchor = np.asarray(q_anchor, dtype=float).reshape(7)
        X_anchor = self._calc_carrier_frame_pose(
            q_anchor,
            q_wsg_instance=self.q_wsg_carry,
        )
        clearance_source_fn = clearance_source_fn or is_free_fn
        R_anchor = X_anchor.rotation()
        p_anchor = np.asarray(X_anchor.translation(), dtype=float).reshape(3)
        approach_axis_W = R_anchor.matrix()[:, 1]
        estimate_clearance_fn = getattr(clearance_source_fn, "estimate_clearance", None)
        target_clearance_m = self._resolve_open_clearance_target(
            clearance_source_fn,
            absolute_clearance_threshold_m,
            open_clearance_margin_m,
        )

        def _check_deadline() -> None:
            self._raise_if_deadline_exceeded(
                planning_deadline_s,
                context="axial_clearance_path",
            )

        def _solve_step(
            pullback_m: float,
            q_seed: np.ndarray,
            step_index: int,
        ) -> np.ndarray:
            _check_deadline()
            X_target = RigidTransform(
                R_anchor,
                p_anchor - float(pullback_m) * approach_axis_W,
            )
            return solve_iiwa_ik_for_gripper_pose(
                plant=self.plant,
                root_context_current=self.root_context_current,
                iiwa_instance=self.iiwa_instance,
                wsg_instance=self.wsg_instance,
                desired_end_effector=X_target,
                q_iiwa_seed=q_seed,
                position_tol=self.options.position_tol,
                theta_tol=self.options.theta_tol,
                max_soft_starts=self.options.local_ik_soft_starts,
                soft_start_sigma=self.options.ik_soft_start_sigma,
                soft_start_random_seed=(
                    self.options.ik_soft_start_seed + soft_start_seed_base + step_index
                ),
                deadline_s=planning_deadline_s,
            )

        def _state_is_free(q: np.ndarray) -> bool:
            _check_deadline()
            return bool(is_free_fn(q))

        def _edge_is_free(q_start: np.ndarray, q_goal: np.ndarray) -> bool:
            _check_deadline()
            return self._strict_edge_is_free(
                q_start,
                q_goal,
                is_free_fn,
                planning_deadline_s=planning_deadline_s,
            )

        def _stop_predicate(q: np.ndarray) -> bool:
            _check_deadline()
            return bool(stop_predicate(q)) if stop_predicate is not None else False

        return build_axial_clearance_path(
            q_anchor=q_anchor,
            step_m=step_m,
            max_pullback_m=max_pullback_m,
            solve_step=_solve_step,
            state_is_free=_state_is_free,
            edge_is_free_fn=_edge_is_free,
            estimate_clearance_fn=estimate_clearance_fn,
            target_clearance_m=target_clearance_m,
            stop_predicate=_stop_predicate if stop_predicate is not None else None,
            check_deadline=_check_deadline,
        )

    @staticmethod
    def _merge_joint_paths(*paths: list[np.ndarray]) -> list[np.ndarray]:
        merged: list[np.ndarray] = []
        for path in paths:
            for q in path:
                q_arr = np.asarray(q, dtype=float).reshape(7).copy()
                if merged and np.linalg.norm(q_arr - merged[-1]) <= 1e-12:
                    continue
                merged.append(q_arr)
        return merged

    def _plan_carry_escape_prefix(
        self,
        q_home: np.ndarray,
        q_postgrasp_retreat: np.ndarray,
        planning_deadline_s: float | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        q_home = np.asarray(q_home, dtype=float).reshape(7)
        q_postgrasp_retreat = np.asarray(q_postgrasp_retreat, dtype=float).reshape(7)
        home_is_carry_feasible = bool(self.is_free_carry(q_home))

        def _can_stage_home(q_current: np.ndarray) -> bool:
            if not bool(self.options.carry_escape_try_home_staging):
                return False
            if not home_is_carry_feasible:
                return False
            return self._strict_edge_is_free(
                q_current,
                q_home,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            )

        if not bool(self.options.enable_carry_escape):
            carry_prefix = [q_postgrasp_retreat.copy()]
        else:
            # If home staging is enabled and feasible, keep pulling back until the
            # carried state can actually connect to home instead of stopping on a
            # merely local clearance threshold.
            absolute_clearance_threshold_m = self.options.carry_escape_clearance_threshold_m
            open_clearance_margin_m = self.options.carry_escape_open_clearance_margin_m
            if bool(self.options.carry_escape_try_home_staging) and home_is_carry_feasible:
                absolute_clearance_threshold_m = None
                open_clearance_margin_m = None
            axial_result = self._plan_axial_clearance_path(
                q_anchor=q_postgrasp_retreat,
                is_free_fn=self.is_free_carry,
                clearance_source_fn=self.is_free_carry,
                absolute_clearance_threshold_m=absolute_clearance_threshold_m,
                open_clearance_margin_m=open_clearance_margin_m,
                max_pullback_m=self.options.carry_escape_max_pullback_m,
                step_m=self.options.axial_pullback_step_m,
                planning_deadline_s=planning_deadline_s,
                stop_predicate=_can_stage_home,
                soft_start_seed_base=1_000,
            )
            carry_prefix = [np.asarray(q, dtype=float).copy() for q in axial_result.path]

        q_carry_start = carry_prefix[-1].copy()
        if _can_stage_home(q_carry_start):
            return self._merge_joint_paths(carry_prefix, [q_home]), q_home.copy()
        return carry_prefix, q_carry_start

    def _find_drop_transit_goal(
        self,
        candidate_index: int,
        X_WG_drop_candidate: RigidTransform,
        q_drop_candidate: np.ndarray,
        planning_deadline_s: float | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        q_drop_candidate = np.asarray(q_drop_candidate, dtype=float).reshape(7)
        default_insertion_path = [q_drop_candidate.copy()]
        if not bool(self.options.enable_drop_preplace):
            return q_drop_candidate.copy(), default_insertion_path

        axial_result = self._plan_axial_clearance_path(
            q_anchor=q_drop_candidate,
            is_free_fn=self.is_free_carry,
            clearance_source_fn=self.drop_preplace_clearance_source,
            absolute_clearance_threshold_m=self.options.drop_preplace_clearance_threshold_m,
            open_clearance_margin_m=self.options.drop_preplace_open_clearance_margin_m,
            max_pullback_m=self.options.drop_preplace_max_pullback_m,
            step_m=self.options.axial_pullback_step_m,
            planning_deadline_s=planning_deadline_s,
            soft_start_seed_base=10_000 + 100 * candidate_index,
        )
        if len(axial_result.path) <= 1:
            return q_drop_candidate.copy(), default_insertion_path

        insertion_path = [
            np.asarray(q, dtype=float).copy() for q in axial_result.path[::-1]
        ]
        q_preplace = insertion_path[0]
        return q_preplace.copy(), insertion_path

    def _iter_drop_transport_bridge_poses(
        self,
        X_WG_drop_candidate: RigidTransform,
        z_offsets_m: tuple[float, ...],
    ):
        R_WG_drop = X_WG_drop_candidate.rotation()
        p_WG_drop = np.asarray(X_WG_drop_candidate.translation(), dtype=float).reshape(3)
        approach_axis_W = R_WG_drop.matrix()[:, 1]
        world_up_W = np.array([0.0, 0.0, 1.0])

        for backoff_m in self.options.drop_transport_bridge_backoff_m:
            for z_offset_m in z_offsets_m:
                p_WG_bridge = (
                    p_WG_drop
                    - float(backoff_m) * approach_axis_W
                    + float(z_offset_m) * world_up_W
                )
                yield float(backoff_m), RigidTransform(R_WG_drop, p_WG_bridge)

    def _solve_carry_pose_ik(
        self,
        X_WG_target: RigidTransform,
        q_seed: np.ndarray,
        soft_start_seed: int,
        planning_deadline_s: float | None = None,
    ) -> np.ndarray:
        q_target = solve_iiwa_ik_for_gripper_pose(
            plant=self.plant,
            root_context_current=self.root_context_current,
            iiwa_instance=self.iiwa_instance,
            wsg_instance=self.wsg_instance,
            desired_end_effector=X_WG_target,
            q_iiwa_seed=q_seed,
            position_tol=self.options.position_tol,
            theta_tol=self.options.theta_tol,
            max_soft_starts=self.options.local_ik_soft_starts,
            soft_start_sigma=self.options.ik_soft_start_sigma,
            soft_start_random_seed=soft_start_seed,
            deadline_s=planning_deadline_s,
        )
        return np.asarray(q_target, dtype=float).reshape(7)

    def _compute_shared_drop_transport_bridges(
        self,
        q_transit_start: np.ndarray,
        X_WG_drop_reference: RigidTransform,
        planning_deadline_s: float | None = None,
    ) -> DropTransportSharedBridges:
        if not bool(self.options.enable_drop_transport_bridges):
            return DropTransportSharedBridges(bridge2_configs=[], bridge1_configs={})

        q_transit_start = np.asarray(q_transit_start, dtype=float).reshape(7)
        shared_bridge2_configs: list[tuple[float, np.ndarray, bool]] = []
        for bridge2_index, (bridge_backoff_m, X_WG_bridge2) in enumerate(
            self._iter_drop_transport_bridge_poses(
                X_WG_drop_reference,
                self.options.drop_transport_bridge_high_z_offset_m,
            )
        ):
            self._raise_if_deadline_exceeded(
                planning_deadline_s,
                context="deterministic_drop_transport",
            )
            try:
                q_bridge2 = self._solve_carry_pose_ik(
                    X_WG_bridge2,
                    q_seed=q_transit_start,
                    soft_start_seed=self.options.ik_soft_start_seed + 20_000 + bridge2_index,
                    planning_deadline_s=planning_deadline_s,
                )
            except Exception:
                continue
            if not self.is_free_carry(q_bridge2):
                continue
            shared_bridge2_configs.append(
                (
                    bridge_backoff_m,
                    q_bridge2.copy(),
                    self._strict_edge_is_free(
                        q_transit_start,
                        q_bridge2,
                        self.is_free_carry,
                        planning_deadline_s=planning_deadline_s,
                    ),
                )
            )

        shared_bridge1_configs: dict[float, list[np.ndarray]] = {}
        for bridge1_index, (bridge_backoff_m, X_WG_bridge1) in enumerate(
            self._iter_drop_transport_bridge_poses(
                X_WG_drop_reference,
                self.options.drop_transport_bridge_low_z_offset_m,
            )
        ):
            self._raise_if_deadline_exceeded(
                planning_deadline_s,
                context="deterministic_drop_transport",
            )
            try:
                q_bridge1 = self._solve_carry_pose_ik(
                    X_WG_bridge1,
                    q_seed=q_transit_start,
                    soft_start_seed=self.options.ik_soft_start_seed + 30_000 + bridge1_index,
                    planning_deadline_s=planning_deadline_s,
                )
            except Exception:
                continue
            if not self.is_free_carry(q_bridge1):
                continue
            if not self._strict_edge_is_free(
                q_transit_start,
                q_bridge1,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            ):
                continue
            shared_bridge1_configs.setdefault(bridge_backoff_m, []).append(q_bridge1.copy())

        return DropTransportSharedBridges(
            bridge2_configs=shared_bridge2_configs,
            bridge1_configs=shared_bridge1_configs,
        )

    def _try_deterministic_drop_transport(
        self,
        q_transit_start: np.ndarray,
        path_carry_escape: list[np.ndarray],
        candidate: tuple[int, RigidTransform, np.ndarray, np.ndarray, list[np.ndarray]],
        shared_bridges: DropTransportSharedBridges,
        planning_deadline_s: float | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], int, RigidTransform] | None:
        if not bool(self.options.enable_drop_transport_bridges):
            return None

        q_transit_start = np.asarray(q_transit_start, dtype=float).reshape(7)
        self._raise_if_deadline_exceeded(
            planning_deadline_s,
            context="deterministic_drop_transport",
        )
        candidate_index, X_WG_drop_candidate, q_drop_candidate, q_transit_goal, insertion_path = candidate
        q_transit_goal = np.asarray(q_transit_goal, dtype=float).reshape(7)

        if self._strict_edge_is_free(
            q_transit_start,
            q_transit_goal,
            self.is_free_carry,
            planning_deadline_s=planning_deadline_s,
        ):
            return (
                np.asarray(q_drop_candidate, dtype=float).copy(),
                self._merge_joint_paths(path_carry_escape, [q_transit_goal], insertion_path),
                int(candidate_index),
                X_WG_drop_candidate,
            )

        for bridge_backoff_m, q_bridge2, start_to_bridge2 in shared_bridges.bridge2_configs:
            if not self._strict_edge_is_free(
                q_bridge2,
                q_transit_goal,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            ):
                continue
            if start_to_bridge2:
                return (
                    np.asarray(q_drop_candidate, dtype=float).copy(),
                    self._merge_joint_paths(
                        path_carry_escape,
                        [q_bridge2, q_transit_goal],
                        insertion_path,
                    ),
                    int(candidate_index),
                    X_WG_drop_candidate,
                )
            for q_bridge1 in shared_bridges.bridge1_configs.get(bridge_backoff_m, []):
                if not self._strict_edge_is_free(
                    q_bridge1,
                    q_bridge2,
                    self.is_free_carry,
                    planning_deadline_s=planning_deadline_s,
                ):
                    continue
                return (
                    np.asarray(q_drop_candidate, dtype=float).copy(),
                    self._merge_joint_paths(
                        path_carry_escape,
                        [q_bridge1, q_bridge2, q_transit_goal],
                        insertion_path,
                    ),
                    int(candidate_index),
                    X_WG_drop_candidate,
                )

        bridge2_cache: dict[float, np.ndarray] = {}
        for bridge2_index, (bridge_backoff_m, X_WG_bridge2) in enumerate(
            self._iter_drop_transport_bridge_poses(
                X_WG_drop_candidate,
                self.options.drop_transport_bridge_high_z_offset_m,
            )
        ):
            self._raise_if_deadline_exceeded(
                planning_deadline_s,
                context="deterministic_drop_transport",
            )
            try:
                q_bridge2 = self._solve_carry_pose_ik(
                    X_WG_bridge2,
                    q_seed=q_transit_goal,
                    soft_start_seed=(
                        self.options.ik_soft_start_seed
                        + 20_000
                        + 200 * candidate_index
                        + bridge2_index
                    ),
                    planning_deadline_s=planning_deadline_s,
                )
            except Exception:
                continue
            if not self.is_free_carry(q_bridge2):
                continue
            if not self._strict_edge_is_free(
                q_bridge2,
                q_transit_goal,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            ):
                continue
            bridge2_cache[bridge_backoff_m] = q_bridge2.copy()

            if self._strict_edge_is_free(
                q_transit_start,
                q_bridge2,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            ):
                return (
                    np.asarray(q_drop_candidate, dtype=float).copy(),
                    self._merge_joint_paths(
                        path_carry_escape,
                        [q_bridge2, q_transit_goal],
                        insertion_path,
                    ),
                    int(candidate_index),
                    X_WG_drop_candidate,
                )

        for bridge1_index, (bridge_backoff_m, X_WG_bridge1) in enumerate(
            self._iter_drop_transport_bridge_poses(
                X_WG_drop_candidate,
                self.options.drop_transport_bridge_low_z_offset_m,
            )
        ):
            if bridge_backoff_m not in bridge2_cache:
                continue
            self._raise_if_deadline_exceeded(
                planning_deadline_s,
                context="deterministic_drop_transport",
            )
            try:
                q_bridge1 = self._solve_carry_pose_ik(
                    X_WG_bridge1,
                    q_seed=q_transit_start,
                    soft_start_seed=(
                        self.options.ik_soft_start_seed
                        + 30_000
                        + 200 * candidate_index
                        + bridge1_index
                    ),
                    planning_deadline_s=planning_deadline_s,
                )
            except Exception:
                continue
            if not self.is_free_carry(q_bridge1):
                continue
            if not self._strict_edge_is_free(
                q_transit_start,
                q_bridge1,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            ):
                continue
            q_bridge2 = bridge2_cache[bridge_backoff_m]
            if not self._strict_edge_is_free(
                q_bridge1,
                q_bridge2,
                self.is_free_carry,
                planning_deadline_s=planning_deadline_s,
            ):
                continue
            return (
                np.asarray(q_drop_candidate, dtype=float).copy(),
                self._merge_joint_paths(
                    path_carry_escape,
                    [q_bridge1, q_bridge2, q_transit_goal],
                    insertion_path,
                ),
                int(candidate_index),
                X_WG_drop_candidate,
            )

        return None

    def run(
        self,
        q_home: np.ndarray,
        X_WG_pregrasp: RigidTransform,
        X_WG_drop: RigidTransform,
    ) -> ManipulationResult:
        self._history = []
        timings_s: dict[str, float] = {}
        planning_start = time.perf_counter()
        planning_deadline_s = None
        if self.options.max_planning_time_s is not None:
            planning_deadline_s = planning_start + float(self.options.max_planning_time_s)

        q_home = np.asarray(q_home, dtype=float).reshape(7)
        self._transition(ManipulationState.HOME_READY)

        try:
            self._transition(ManipulationState.PLAN_HOME_TO_PREGRASP)
            phase_start = time.perf_counter()
            home_deadline_s = self._resolve_stage_deadline(
                planning_deadline_s,
                stage_start_s=phase_start,
                stage_budget_s=self.options.home_to_pregrasp_time_budget_s,
            )
            self._check_stage_deadline(
                "home_to_pregrasp",
                planning_deadline_s=planning_deadline_s,
                stage_deadline_s=home_deadline_s,
                stage_budget_s=self.options.home_to_pregrasp_time_budget_s,
            )
            q_pregrasp = self._require_pose_reachable(
                pose_label="Pickup",
                X_WG_target=X_WG_pregrasp,
                q_seed=q_home,
                planning_deadline_s=home_deadline_s,
                soft_start_seed_offset=0,
            )
            if not self.is_free(q_pregrasp):
                raise RuntimeError("Pregrasp IK solution is not collision free")
            path_home_to_pregrasp = self._plan_rrt(
                q_home,
                q_pregrasp,
                search_is_free_fn=self.is_free,
                planning_deadline_s=home_deadline_s,
                random_seed=self._rrt_random_seed(1),
                debug_label="home_to_pregrasp",
            )
            timings_s["plan_home_to_pregrasp"] = time.perf_counter() - phase_start

            self._transition(ManipulationState.RUN_GRASP_PRIMITIVE)
            phase_start = time.perf_counter()
            grasp_deadline_s = self._resolve_stage_deadline(
                planning_deadline_s,
                stage_start_s=phase_start,
                stage_budget_s=self.options.grasp_primitive_time_budget_s,
            )
            self._check_stage_deadline(
                "grasp_primitive",
                planning_deadline_s=planning_deadline_s,
                stage_deadline_s=grasp_deadline_s,
                stage_budget_s=self.options.grasp_primitive_time_budget_s,
            )
            grasp_result = plan_grasp_primitive(
                plant=self.plant,
                root_context_current=self.root_context_current,
                iiwa_instance=self.iiwa_instance,
                wsg_instance=self.wsg_instance,
                is_free=self.is_free_grasp,
                joints_lower_limits=self.joints_lower_limits,
                joints_upper_limits=self.joints_upper_limits,
                q_pregrasp=q_pregrasp,
                X_WG_pregrasp=X_WG_pregrasp,
                planner_config=self.options.rrt,
                options=self.options.grasp_options,
                is_free_retreat=self.is_free_deapproach,
                prepare_retreat_checker=self._configure_carry_payload_attachment,
                planning_deadline_s=grasp_deadline_s,
            )
            timings_s["plan_grasp_primitive"] = time.perf_counter() - phase_start
            if not grasp_result.success or grasp_result.plan is None:
                details = "; ".join(grasp_result.failure_reasons)
                raise RuntimeError(f"Grasp primitive failed. {details}")
            grasp_plan = grasp_result.plan

            # After grasp, treat payload as rigidly attached for carry/drop checks.
            self._configure_carry_payload_attachment(grasp_plan.q_grasp)

            self._transition(ManipulationState.PLAN_PREGRASP_TO_DROP)
            phase_start = time.perf_counter()
            drop_deadline_s = self._resolve_stage_deadline(
                planning_deadline_s,
                stage_start_s=phase_start,
                stage_budget_s=self.options.pregrasp_to_drop_time_budget_s,
            )
            self._check_stage_deadline(
                "pregrasp_to_drop",
                planning_deadline_s=planning_deadline_s,
                stage_deadline_s=drop_deadline_s,
                stage_budget_s=self.options.pregrasp_to_drop_time_budget_s,
            )
            self._require_pose_reachable(
                pose_label="Drop",
                X_WG_target=X_WG_drop,
                q_seed=grasp_plan.q_postgrasp_retreat,
                planning_deadline_s=drop_deadline_s,
                soft_start_seed_offset=100,
            )
            carry_escape_start = time.perf_counter()
            path_carry_escape, q_drop_rrt_start = self._plan_carry_escape_prefix(
                q_home=q_home,
                q_postgrasp_retreat=grasp_plan.q_postgrasp_retreat,
                planning_deadline_s=drop_deadline_s,
            )
            timings_s["plan_carry_escape"] = time.perf_counter() - carry_escape_start
            q_drop = None
            path_pregrasp_to_drop = None
            drop_candidate_index = -1
            X_WG_drop_selected = None
            drop_failures = []
            drop_candidates_tried = 0
            drop_feasible_candidates = 0
            drop_anchor_ik_time_s = 0.0
            drop_preplace_time_s = 0.0
            drop_rrt_time_s = 0.0
            drop_rrt_calls = 0
            feasible_drop_candidates: list[
                tuple[int, RigidTransform, np.ndarray, np.ndarray, list[np.ndarray]]
            ] = []
            drop_rrt_batch_sizes = tuple(
                max(1, int(batch_size))
                for batch_size in self.options.drop_rrt_candidate_batch_sizes
            )
            next_drop_rrt_batch_index = 0

            def _resolve_drop_candidate_deadline() -> float | None:
                candidate_deadline_s = drop_deadline_s
                if self.options.drop_candidate_time_budget_s is not None:
                    local_deadline_s = (
                        time.perf_counter() + float(self.options.drop_candidate_time_budget_s)
                    )
                    if candidate_deadline_s is None:
                        candidate_deadline_s = local_deadline_s
                    else:
                        candidate_deadline_s = min(candidate_deadline_s, local_deadline_s)
                return candidate_deadline_s

            def _attempt_drop_rrt(force: bool = False) -> bool:
                nonlocal next_drop_rrt_batch_index
                nonlocal q_drop
                nonlocal path_pregrasp_to_drop
                nonlocal drop_candidate_index
                nonlocal X_WG_drop_selected
                nonlocal drop_rrt_time_s
                nonlocal drop_rrt_calls

                if not feasible_drop_candidates:
                    return False

                while next_drop_rrt_batch_index < len(drop_rrt_batch_sizes):
                    batch_size = min(
                        int(drop_rrt_batch_sizes[next_drop_rrt_batch_index]),
                        len(feasible_drop_candidates),
                    )
                    if not force and len(feasible_drop_candidates) < int(
                        drop_rrt_batch_sizes[next_drop_rrt_batch_index]
                    ):
                        return False

                    q_drop_candidates = [
                        candidate[3] for candidate in feasible_drop_candidates[:batch_size]
                    ]
                    seed_offsets = self._ordered_drop_rrt_seed_offsets(batch_size)
                    for retry_index, seed_offset in enumerate(seed_offsets):
                        candidate_deadline_s = _resolve_drop_candidate_deadline()
                        rrt_batch_seed = self._rrt_random_seed(seed_offset)
                        rrt_batch_start = time.perf_counter()
                        drop_rrt_calls += 1
                        try:
                            path_candidate, selected_goal_slot = self._plan_rrt(
                                q_drop_rrt_start,
                                q_drop_candidates,
                                is_free_fn=self.is_free_carry,
                                search_is_free_fn=self.is_free_carry,
                                planning_deadline_s=candidate_deadline_s,
                                random_seed=rrt_batch_seed,
                                return_goal_index=True,
                                debug_label=(
                                    "drop_candidates_multi_goal_batch_"
                                    f"{batch_size}_retry_{retry_index}"
                                ),
                            )
                        except TimeoutError as exc:
                            drop_failures.append(
                                "drop_candidates_multi_goal_batch_"
                                f"{batch_size}_retry_{retry_index}: {exc}"
                            )
                        except Exception as exc:
                            drop_failures.append(
                                "drop_candidates_multi_goal_batch_"
                                f"{batch_size}_retry_{retry_index}: {exc}"
                            )
                        else:
                            (
                                selected_candidate_index,
                                selected_drop_pose,
                                selected_q_drop,
                                _selected_transit_goal,
                                selected_insertion_path,
                            ) = feasible_drop_candidates[int(selected_goal_slot)]
                            q_drop = np.asarray(selected_q_drop, dtype=float).copy()
                            path_pregrasp_to_drop = self._merge_joint_paths(
                                path_carry_escape,
                                path_candidate,
                                selected_insertion_path,
                            )
                            drop_candidate_index = int(selected_candidate_index)
                            X_WG_drop_selected = selected_drop_pose
                            next_drop_rrt_batch_index += 1
                            return True
                        finally:
                            drop_rrt_time_s += time.perf_counter() - rrt_batch_start

                    next_drop_rrt_batch_index += 1
                    if not force:
                        return False

                return False

            for candidate_index, X_WG_drop_candidate in self._iter_drop_candidates(X_WG_drop):
                self._check_stage_deadline(
                    "pregrasp_to_drop",
                    planning_deadline_s=planning_deadline_s,
                    stage_deadline_s=drop_deadline_s,
                    stage_budget_s=self.options.pregrasp_to_drop_time_budget_s,
                )
                if (
                    self.options.max_drop_candidates is not None
                    and drop_candidates_tried >= int(self.options.max_drop_candidates)
                ):
                    break
                drop_candidates_tried += 1
                try:
                    anchor_ik_start = time.perf_counter()
                    try:
                        q_drop_candidate = solve_iiwa_ik_for_gripper_pose(
                            plant=self.plant,
                            root_context_current=self.root_context_current,
                            iiwa_instance=self.iiwa_instance,
                            wsg_instance=self.wsg_instance,
                            desired_end_effector=X_WG_drop_candidate,
                            q_iiwa_seed=grasp_plan.q_postgrasp_retreat,
                            position_tol=self.options.position_tol,
                            theta_tol=self.options.theta_tol,
                            max_soft_starts=self.options.ik_soft_starts,
                            soft_start_sigma=self.options.ik_soft_start_sigma,
                            soft_start_random_seed=self.options.ik_soft_start_seed + 100 + candidate_index,
                            deadline_s=drop_deadline_s,
                        )
                    finally:
                        drop_anchor_ik_time_s += time.perf_counter() - anchor_ik_start
                    if not self.is_free_carry(q_drop_candidate):
                        drop_failures.append(
                            f"drop_candidate_{candidate_index}: IK solution not collision free for carry state"
                        )
                        continue
                    preplace_start = time.perf_counter()
                    try:
                        q_drop_transit_goal, insertion_path = self._find_drop_transit_goal(
                            candidate_index=candidate_index,
                            X_WG_drop_candidate=X_WG_drop_candidate,
                            q_drop_candidate=q_drop_candidate,
                            planning_deadline_s=drop_deadline_s,
                        )
                    finally:
                        drop_preplace_time_s += time.perf_counter() - preplace_start
                    candidate = (
                        candidate_index,
                        X_WG_drop_candidate,
                        np.asarray(q_drop_candidate, dtype=float).copy(),
                        np.asarray(q_drop_transit_goal, dtype=float).copy(),
                        [np.asarray(q, dtype=float).copy() for q in insertion_path],
                    )
                    feasible_drop_candidates.append(candidate)
                    drop_feasible_candidates += 1
                    # Professional transport stack:
                    # carry escape -> seeded RRT to pre-place -> deterministic insertion.
                    if _attempt_drop_rrt(force=False):
                        break
                except TimeoutError:
                    drop_failures.append(f"drop_candidate_{candidate_index}: planning timeout")
                except Exception as exc:
                    drop_failures.append(f"drop_candidate_{candidate_index}: {exc}")

            if q_drop is None and feasible_drop_candidates:
                _attempt_drop_rrt(force=True)

            timings_s["plan_drop_anchor_ik"] = drop_anchor_ik_time_s
            timings_s["plan_drop_preplace"] = drop_preplace_time_s
            timings_s["plan_drop_rrt"] = drop_rrt_time_s
            timings_s["plan_pregrasp_to_drop"] = time.perf_counter() - phase_start
            timings_s["drop_candidates_tried"] = float(drop_candidates_tried)
            timings_s["drop_feasible_candidates"] = float(drop_feasible_candidates)
            timings_s["drop_rrt_calls"] = float(drop_rrt_calls)
            if q_drop is None or path_pregrasp_to_drop is None or X_WG_drop_selected is None:
                details = "; ".join(drop_failures)
                raise RuntimeError(f"Drop planning failed. {details}")

            self._transition(ManipulationState.RELEASE_AT_DROP)
            self._transition(ManipulationState.DONE)
            timings_s["plan_total"] = time.perf_counter() - planning_start
            return ManipulationResult(
                success=True,
                final_state=ManipulationState.DONE,
                plan=ManipulationPlan(
                    q_home=q_home.copy(),
                    q_pregrasp=np.asarray(q_pregrasp, dtype=float).copy(),
                    q_grasp=np.asarray(grasp_plan.q_grasp, dtype=float).copy(),
                    q_drop=np.asarray(q_drop, dtype=float).copy(),
                    drop_candidate_index=drop_candidate_index,
                    X_WG_drop_selected=X_WG_drop_selected,
                    path_home_to_pregrasp=[np.asarray(q, dtype=float).copy() for q in path_home_to_pregrasp],
                    grasp_plan=grasp_plan,
                    path_pregrasp_to_drop=[np.asarray(q, dtype=float).copy() for q in path_pregrasp_to_drop],
                    gripper_events=[
                        "close_before_home_to_pregrasp",
                        "open_at_pregrasp_before_grasp",
                        "close_at_grasp",
                        "open_at_drop",
                    ],
                ),
                error_message=None,
                transition_history=self._history.copy(),
                timings_s=timings_s.copy(),
            )
        except Exception as exc:
            self._transition(ManipulationState.FAILED)
            timings_s["plan_total"] = time.perf_counter() - planning_start
            return ManipulationResult(
                success=False,
                final_state=ManipulationState.FAILED,
                plan=None,
                error_message=str(exc),
                transition_history=self._history.copy(),
                timings_s=timings_s.copy(),
            )
