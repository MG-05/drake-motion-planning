import dataclasses as dc
import enum
import typing

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from src.manipulation.grasp import GraspOptions, GraspPrimitivePlan, plan_grasp_primitive
from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt_connect import rrt_connect_plan


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
    position_tol: float = 0.001
    theta_tol: float = 0.035
    ik_soft_starts: int = 20
    ik_soft_start_sigma: float = 0.08
    ik_soft_start_seed: int = 0
    rrt_step_size: float = 0.12
    rrt_goal_sample_rate: float = 0.20
    rrt_max_iters: int = 50_000
    rrt_edge_resolution: float = 0.005
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
        configure_fn = getattr(self.is_free_carry, "configure_attached_model", None)
        if not callable(configure_fn):
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
    ) -> list[np.ndarray]:
        is_free_fn = is_free_fn or self.is_free
        return rrt_connect_plan(
            q_start=q_start,
            q_goal=q_goal,
            is_free=is_free_fn,
            joints_lower_limits=self.joints_lower_limits,
            joints_upper_limits=self.joints_upper_limits,
            step_size=self.options.rrt_step_size,
            goal_sample_rate=self.options.rrt_goal_sample_rate,
            max_iters=self.options.rrt_max_iters,
            edge_resolution=self.options.rrt_edge_resolution,
            enable_shortcut=True,
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
        q_home = np.asarray(q_home, dtype=float).reshape(7)
        self._transition(ManipulationState.HOME_READY)

        try:
            self._transition(ManipulationState.PLAN_HOME_TO_PREGRASP)
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
            path_home_to_pregrasp = self._plan_rrt(q_home, q_pregrasp)

            self._transition(ManipulationState.RUN_GRASP_PRIMITIVE)
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
                options=self.options.grasp_options,
            )
            if not grasp_result.success or grasp_result.plan is None:
                details = "; ".join(grasp_result.failure_reasons)
                raise RuntimeError(f"Grasp primitive failed. {details}")
            grasp_plan = grasp_result.plan

            # After grasp, treat payload as rigidly attached for carry/drop checks.
            self._configure_carry_payload_attachment(grasp_plan.q_grasp)

            self._transition(ManipulationState.PLAN_PREGRASP_TO_DROP)
            q_drop = None
            path_pregrasp_to_drop = None
            drop_candidate_index = -1
            X_WG_drop_selected = None
            drop_failures = []
            for candidate_index, X_WG_drop_candidate in self._iter_drop_candidates(X_WG_drop):
                try:
                    q_drop_candidate = solve_iiwa_ik_for_gripper_pose(
                        plant=self.plant,
                        root_context_current=self.root_context_current,
                        iiwa_instance=self.iiwa_instance,
                        wsg_instance=self.wsg_instance,
                        desired_end_effector=X_WG_drop_candidate,
                        q_iiwa_seed=grasp_plan.q_pregrasp,
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
                    path_candidate = self._plan_rrt(
                        grasp_plan.q_pregrasp,
                        q_drop_candidate,
                        is_free_fn=self.is_free_carry,
                    )
                    q_drop = np.asarray(q_drop_candidate, dtype=float).copy()
                    path_pregrasp_to_drop = path_candidate
                    drop_candidate_index = candidate_index
                    X_WG_drop_selected = X_WG_drop_candidate
                    break
                except Exception as exc:
                    drop_failures.append(f"drop_candidate_{candidate_index}: {exc}")

            if q_drop is None or path_pregrasp_to_drop is None or X_WG_drop_selected is None:
                details = "; ".join(drop_failures)
                raise RuntimeError(f"Drop planning failed. {details}")

            self._transition(ManipulationState.RELEASE_AT_DROP)
            self._transition(ManipulationState.DONE)
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
            )
        except Exception as exc:
            self._transition(ManipulationState.FAILED)
            return ManipulationResult(
                success=False,
                final_state=ManipulationState.FAILED,
                plan=None,
                error_message=str(exc),
                transition_history=self._history.copy(),
            )
