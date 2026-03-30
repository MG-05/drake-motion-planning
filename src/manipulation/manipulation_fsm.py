import dataclasses as dc
import enum
import time
import typing

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from src.manipulation.grasp import GraspOptions, GraspPrimitivePlan, plan_grasp_primitive
from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt_connect import RRTConnectConfig, edge_is_free, rrt_connect_plan


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
    ik_soft_start_sigma: float = 0.08
    ik_soft_start_seed: int = 0
    rrt: RRTConnectConfig = dc.field(default_factory=RRTConnectConfig)
    max_planning_time_s: float | None = 60.0
    drop_candidate_time_budget_s: float | None = 8.0
    max_drop_candidates: int | None = 20
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
    enable_drop_transport_bridges: bool = True
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
    drop_yaw_offsets_rad: tuple[float, ...] = (0.0, 0.20, -0.20)
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
        for checker in (self.is_free_deapproach, self.is_free_carry):
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

    def _plan_rrt(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        is_free_fn: typing.Callable[[np.ndarray], bool] | None = None,
        planning_deadline_s: float | None = None,
        return_goal_index: bool = False,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], int]:
        is_free_fn = is_free_fn or self.is_free
        return rrt_connect_plan(
            q_start=q_start,
            q_goal=q_goal,
            is_free=is_free_fn,
            joints_lower_limits=self.joints_lower_limits,
            joints_upper_limits=self.joints_upper_limits,
            **self.options.rrt.to_plan_kwargs(deadline_s=planning_deadline_s),
            return_goal_index=return_goal_index,
        )

    def _iter_drop_candidates(self, X_WG_drop: RigidTransform):
        p_WG_nominal = X_WG_drop.translation()
        R_WG_nominal = X_WG_drop.rotation()

        candidate_index = 0
        for dx, dy in self.options.drop_xy_offsets_m:
            for dz in self.options.drop_z_offsets_m:
                for yaw in self.options.drop_yaw_offsets_rad:
                    if abs(float(yaw)) <= 1e-12:
                        R_WG_candidate = R_WG_nominal
                    else:
                        R_WYaw = RollPitchYaw(0.0, 0.0, float(yaw)).ToRotationMatrix()
                        R_WG_candidate = RotationMatrix(R_WYaw.matrix() @ R_WG_nominal.matrix())
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
        R_anchor = X_anchor.rotation()
        p_anchor = np.asarray(X_anchor.translation(), dtype=float).reshape(3)
        approach_axis_W = R_anchor.matrix()[:, 1]
        estimate_clearance_fn = getattr(is_free_fn, "estimate_clearance", None)
        target_clearance_m = self._resolve_open_clearance_target(
            is_free_fn,
            absolute_clearance_threshold_m,
            open_clearance_margin_m,
        )

        def _check_deadline() -> None:
            if planning_deadline_s is not None and time.perf_counter() > float(
                planning_deadline_s
            ):
                raise TimeoutError(
                    f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s"
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
                max_soft_starts=self.options.ik_soft_starts,
                soft_start_sigma=self.options.ik_soft_start_sigma,
                soft_start_random_seed=(
                    self.options.ik_soft_start_seed + soft_start_seed_base + step_index
                ),
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
            axial_result = self._plan_axial_clearance_path(
                q_anchor=q_postgrasp_retreat,
                is_free_fn=self.is_free_carry,
                absolute_clearance_threshold_m=self.options.carry_escape_clearance_threshold_m,
                open_clearance_margin_m=self.options.carry_escape_open_clearance_margin_m,
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
            max_soft_starts=self.options.ik_soft_starts,
            soft_start_sigma=self.options.ik_soft_start_sigma,
            soft_start_random_seed=soft_start_seed,
        )
        return np.asarray(q_target, dtype=float).reshape(7)

    def _try_deterministic_drop_transport(
        self,
        q_transit_start: np.ndarray,
        path_carry_escape: list[np.ndarray],
        X_WG_drop_reference: RigidTransform,
        feasible_drop_candidates: list[tuple[int, RigidTransform, np.ndarray, np.ndarray, list[np.ndarray]]],
        planning_deadline_s: float | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], int, RigidTransform] | None:
        if not bool(self.options.enable_drop_transport_bridges):
            return None

        q_transit_start = np.asarray(q_transit_start, dtype=float).reshape(7)
        shared_bridge2_configs: list[tuple[float, np.ndarray, bool]] = []
        for bridge2_index, (bridge_backoff_m, X_WG_bridge2) in enumerate(
            self._iter_drop_transport_bridge_poses(
                X_WG_drop_reference,
                self.options.drop_transport_bridge_high_z_offset_m,
            )
        ):
            if planning_deadline_s is not None and time.perf_counter() > float(planning_deadline_s):
                raise TimeoutError(
                    f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s"
                )
            try:
                q_bridge2 = self._solve_carry_pose_ik(
                    X_WG_bridge2,
                    q_seed=q_transit_start,
                    soft_start_seed=self.options.ik_soft_start_seed + 20_000 + bridge2_index,
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
            if planning_deadline_s is not None and time.perf_counter() > float(planning_deadline_s):
                raise TimeoutError(
                    f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s"
                )
            try:
                q_bridge1 = self._solve_carry_pose_ik(
                    X_WG_bridge1,
                    q_seed=q_transit_start,
                    soft_start_seed=self.options.ik_soft_start_seed + 30_000 + bridge1_index,
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

        for goal_slot, candidate in enumerate(feasible_drop_candidates):
            if planning_deadline_s is not None and time.perf_counter() > float(planning_deadline_s):
                raise TimeoutError(
                    f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s"
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

            for bridge_backoff_m, q_bridge2, start_to_bridge2 in shared_bridge2_configs:
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
                for q_bridge1 in shared_bridge1_configs.get(bridge_backoff_m, []):
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
                if planning_deadline_s is not None and time.perf_counter() > float(planning_deadline_s):
                    raise TimeoutError(
                        f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s"
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
                if planning_deadline_s is not None and time.perf_counter() > float(planning_deadline_s):
                    raise TimeoutError(
                        f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s"
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

        def _check_planning_deadline() -> None:
            if planning_deadline_s is not None and time.perf_counter() > planning_deadline_s:
                raise TimeoutError(f"Planning exceeded time budget of {self.options.max_planning_time_s:.1f}s")

        q_home = np.asarray(q_home, dtype=float).reshape(7)
        self._transition(ManipulationState.HOME_READY)

        try:
            self._transition(ManipulationState.PLAN_HOME_TO_PREGRASP)
            phase_start = time.perf_counter()
            _check_planning_deadline()
            q_pregrasp = solve_iiwa_ik_for_gripper_pose(
                plant=self.plant,
                root_context_current=self.root_context_current,
                iiwa_instance=self.iiwa_instance,
                wsg_instance=self.wsg_instance,
                desired_end_effector=X_WG_pregrasp,
                q_iiwa_seed=q_home,
                position_tol=self.options.position_tol,
                theta_tol=self.options.theta_tol,
                max_soft_starts=self.options.ik_soft_starts,
                soft_start_sigma=self.options.ik_soft_start_sigma,
                soft_start_random_seed=self.options.ik_soft_start_seed,
            )
            if not self.is_free(q_pregrasp):
                raise RuntimeError("Pregrasp IK solution is not collision free")
            path_home_to_pregrasp = self._plan_rrt(
                q_home, q_pregrasp, planning_deadline_s=planning_deadline_s
            )
            timings_s["plan_home_to_pregrasp"] = time.perf_counter() - phase_start

            self._transition(ManipulationState.RUN_GRASP_PRIMITIVE)
            phase_start = time.perf_counter()
            _check_planning_deadline()
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
                planning_deadline_s=planning_deadline_s,
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
            carry_escape_start = time.perf_counter()
            path_carry_escape, q_drop_rrt_start = self._plan_carry_escape_prefix(
                q_home=q_home,
                q_postgrasp_retreat=grasp_plan.q_postgrasp_retreat,
                planning_deadline_s=planning_deadline_s,
            )
            timings_s["plan_carry_escape"] = time.perf_counter() - carry_escape_start
            q_drop = None
            path_pregrasp_to_drop = None
            drop_candidate_index = -1
            X_WG_drop_selected = None
            drop_failures = []
            drop_candidates_tried = 0
            feasible_drop_candidates: list[
                tuple[int, RigidTransform, np.ndarray, np.ndarray, list[np.ndarray]]
            ] = []
            for candidate_index, X_WG_drop_candidate in self._iter_drop_candidates(X_WG_drop):
                _check_planning_deadline()
                if (
                    self.options.max_drop_candidates is not None
                    and drop_candidates_tried >= int(self.options.max_drop_candidates)
                ):
                    break
                drop_candidates_tried += 1
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
                    )
                    if not self.is_free_carry(q_drop_candidate):
                        drop_failures.append(
                            f"drop_candidate_{candidate_index}: IK solution not collision free for carry state"
                        )
                        continue
                    q_drop_transit_goal, insertion_path = self._find_drop_transit_goal(
                        candidate_index=candidate_index,
                        X_WG_drop_candidate=X_WG_drop_candidate,
                        q_drop_candidate=q_drop_candidate,
                        planning_deadline_s=planning_deadline_s,
                    )
                    feasible_drop_candidates.append(
                        (
                            candidate_index,
                            X_WG_drop_candidate,
                            np.asarray(q_drop_candidate, dtype=float).copy(),
                            np.asarray(q_drop_transit_goal, dtype=float).copy(),
                            [np.asarray(q, dtype=float).copy() for q in insertion_path],
                        )
                    )
                    deterministic_drop = self._try_deterministic_drop_transport(
                        q_transit_start=q_drop_rrt_start,
                        path_carry_escape=path_carry_escape,
                        X_WG_drop_reference=X_WG_drop,
                        feasible_drop_candidates=feasible_drop_candidates,
                        planning_deadline_s=planning_deadline_s,
                    )
                    if deterministic_drop is not None:
                        (
                            q_drop,
                            path_pregrasp_to_drop,
                            drop_candidate_index,
                            X_WG_drop_selected,
                        ) = deterministic_drop
                        break
                except TimeoutError:
                    drop_failures.append(f"drop_candidate_{candidate_index}: planning timeout")
                except Exception as exc:
                    drop_failures.append(f"drop_candidate_{candidate_index}: {exc}")

            if q_drop is None and feasible_drop_candidates:
                candidate_deadline_s = planning_deadline_s
                if self.options.drop_candidate_time_budget_s is not None:
                    local_deadline_s = (
                        time.perf_counter() + float(self.options.drop_candidate_time_budget_s)
                    )
                    if candidate_deadline_s is None:
                        candidate_deadline_s = local_deadline_s
                    else:
                        candidate_deadline_s = min(candidate_deadline_s, local_deadline_s)
                try:
                    deterministic_drop = self._try_deterministic_drop_transport(
                        q_transit_start=q_drop_rrt_start,
                        path_carry_escape=path_carry_escape,
                        X_WG_drop_reference=X_WG_drop,
                        feasible_drop_candidates=feasible_drop_candidates,
                        planning_deadline_s=candidate_deadline_s,
                    )
                    if deterministic_drop is not None:
                        (
                            q_drop,
                            path_pregrasp_to_drop,
                            drop_candidate_index,
                            X_WG_drop_selected,
                        ) = deterministic_drop
                    else:
                        q_drop_candidates = [candidate[3] for candidate in feasible_drop_candidates]
                        path_candidate, selected_goal_slot = self._plan_rrt(
                            q_drop_rrt_start,
                            q_drop_candidates,
                            is_free_fn=self.is_free_carry,
                            planning_deadline_s=candidate_deadline_s,
                            return_goal_index=True,
                        )
                        (
                            selected_candidate_index,
                            selected_drop_pose,
                            selected_q_drop,
                            _selected_transit_goal,
                            selected_insertion_path,
                        ) = feasible_drop_candidates[
                            int(selected_goal_slot)
                        ]
                        q_drop = np.asarray(selected_q_drop, dtype=float).copy()
                        path_pregrasp_to_drop = self._merge_joint_paths(
                            path_carry_escape,
                            path_candidate,
                            selected_insertion_path,
                        )
                        drop_candidate_index = int(selected_candidate_index)
                        X_WG_drop_selected = selected_drop_pose
                except TimeoutError:
                    drop_failures.append("drop_candidates_multi_goal: RRT timeout")
                except Exception as exc:
                    drop_failures.append(f"drop_candidates_multi_goal: {exc}")

            timings_s["plan_pregrasp_to_drop"] = time.perf_counter() - phase_start
            timings_s["drop_candidates_tried"] = float(drop_candidates_tried)
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
