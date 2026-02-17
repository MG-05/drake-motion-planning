import dataclasses as dc
import typing

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.rrt_connect import rrt_connect_plan


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
    rrt_step_size: float = 0.12
    rrt_goal_sample_rate: float = 0.20
    rrt_max_iters: int = 50000
    rrt_edge_resolution: float = 0.05
    variants: tuple[GraspVariant, ...] = (
        GraspVariant(approach_depth_m=0.1, lateral_offset_m=0.00, yaw_offset_rad=0.00),
        GraspVariant(approach_depth_m=0.09, lateral_offset_m=0.01, yaw_offset_rad=0.00),
        GraspVariant(approach_depth_m=0.11, lateral_offset_m=-0.01, yaw_offset_rad=0.00),
        GraspVariant(approach_depth_m=0.09, lateral_offset_m=0.00, yaw_offset_rad=0.10),
        GraspVariant(approach_depth_m=0.11, lateral_offset_m=0.00, yaw_offset_rad=-0.10),
    )


@dc.dataclass
class GraspPrimitivePlan:
    """
    Planned motion for pregrasp -> grasp -> pregrasp.
    """
    selected_variant_index: int
    selected_variant: GraspVariant
    X_WG_grasp: RigidTransform
    q_pregrasp: np.ndarray
    q_grasp: np.ndarray
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
    options: GraspOptions,
) -> list[np.ndarray]:
    return rrt_connect_plan(
        q_start=q_start,
        q_goal=q_goal,
        is_free=is_free,
        joints_lower_limits=joints_lower_limits,
        joints_upper_limits=joints_upper_limits,
        step_size=options.rrt_step_size,
        goal_sample_rate=options.rrt_goal_sample_rate,
        max_iters=options.rrt_max_iters,
        edge_resolution=options.rrt_edge_resolution,
    )


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
    options: GraspOptions | None = None,
) -> GraspPrimitiveResult:
    """
    Plans the grasp primitive only:
      pregrasp -> grasp -> pregrasp
    """
    options = options or GraspOptions()
    q_pregrasp = np.asarray(q_pregrasp, dtype=float).reshape(7)
    joints_lower_limits = np.asarray(joints_lower_limits, dtype=float).reshape(7)
    joints_upper_limits = np.asarray(joints_upper_limits, dtype=float).reshape(7)

    failure_reasons: list[str] = []

    for variant_index, variant in enumerate(options.variants):
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
            )

            if not is_free(q_grasp):
                failure_reasons.append(f"variant_{variant_index}: IK solution not collision-free")
                continue

            path_pregrasp_to_grasp = _plan_rrt_segment(
                q_start=q_pregrasp,
                q_goal=q_grasp,
                is_free=is_free,
                joints_lower_limits=joints_lower_limits,
                joints_upper_limits=joints_upper_limits,
                options=options,
            )

            path_grasp_to_pregrasp = _plan_rrt_segment(
                q_start=q_grasp,
                q_goal=q_pregrasp,
                is_free=is_free,
                joints_lower_limits=joints_lower_limits,
                joints_upper_limits=joints_upper_limits,
                options=options,
            )

            return GraspPrimitiveResult(
                success=True,
                plan=GraspPrimitivePlan(
                    selected_variant_index=variant_index,
                    selected_variant=variant,
                    X_WG_grasp=X_WG_grasp,
                    q_pregrasp=q_pregrasp.copy(),
                    q_grasp=np.asarray(q_grasp, dtype=float).copy(),
                    path_pregrasp_to_grasp=[np.asarray(q, dtype=float).copy() for q in path_pregrasp_to_grasp],
                    path_grasp_to_pregrasp=[np.asarray(q, dtype=float).copy() for q in path_grasp_to_pregrasp],
                ),
                failure_reasons=failure_reasons,
            )
        except Exception as exc:
            failure_reasons.append(f"variant_{variant_index}: {exc}")

    return GraspPrimitiveResult(success=False, plan=None, failure_reasons=failure_reasons)
