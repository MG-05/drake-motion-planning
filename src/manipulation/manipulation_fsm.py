import dataclasses as dc
import enum
import time
import typing

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from src.manipulation.grasp import GraspOptions, GraspPrimitivePlan, plan_grasp_primitive
from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt_connect import RRTConnectConfig, rrt_connect_plan


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
    ) -> list[np.ndarray]:
        is_free_fn = is_free_fn or self.is_free
        return rrt_connect_plan(
            q_start=q_start,
            q_goal=q_goal,
            is_free=is_free_fn,
            joints_lower_limits=self.joints_lower_limits,
            joints_upper_limits=self.joints_upper_limits,
            **self.options.rrt.to_plan_kwargs(deadline_s=planning_deadline_s),
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
            q_drop = None
            path_pregrasp_to_drop = None
            drop_candidate_index = -1
            X_WG_drop_selected = None
            drop_failures = []
            drop_candidates_tried = 0
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
                    candidate_deadline_s = planning_deadline_s
                    if self.options.drop_candidate_time_budget_s is not None:
                        local_deadline_s = (
                            time.perf_counter() + float(self.options.drop_candidate_time_budget_s)
                        )
                        if candidate_deadline_s is None:
                            candidate_deadline_s = local_deadline_s
                        else:
                            candidate_deadline_s = min(candidate_deadline_s, local_deadline_s)
                    path_candidate = self._plan_rrt(
                        grasp_plan.q_postgrasp_retreat,
                        q_drop_candidate,
                        is_free_fn=self.is_free_carry,
                        planning_deadline_s=candidate_deadline_s,
                    )
                    q_drop = np.asarray(q_drop_candidate, dtype=float).copy()
                    path_pregrasp_to_drop = path_candidate
                    drop_candidate_index = candidate_index
                    X_WG_drop_selected = X_WG_drop_candidate
                    break
                except TimeoutError:
                    drop_failures.append(f"drop_candidate_{candidate_index}: RRT timeout")
                except Exception as exc:
                    drop_failures.append(f"drop_candidate_{candidate_index}: {exc}")

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
