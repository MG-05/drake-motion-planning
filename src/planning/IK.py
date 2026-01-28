import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.solvers import Solve


def solve_iiwa_ik_for_gripper_pose(
        plant,
        plant_context_current,
        iiwa_instance,
        wsg_instance,
        desired_end_effector,
        q_iiwa_seed,
        position_tol=0.01,
        theta_tol=0.01
):
    """
    Returns the q_goal for iiwa that achieves the desired gripper pose desired_end_effector
    """

    # Create a fresh context and copy plant into it
    ik_context = plant.CreateDefaultContext()
    q0_all = plant.GetPositions(plant_context_current)
    plant.SetPositions(ik_context, q0_all)

    # Lock gripper finger joints
    for finger_name in ["left_finger_sliding_joint", "right_finger_sliding_joint"]:
        joint = plant.GetJointByName(finger_name, wsg_instance)
        joint.Lock(ik_context)

    # lock free floating objects (for when I add back the foam brick)
    if plant.HasModelInstanceNamed("foam_brick"):
        brick = plant.GetModelInstanceByName("foam_brick")
        for joint_index in plant.GetJointIndices(brick):
            plant.get_joint(joint_index).Lock(ik_context)

    # Build IK using the context. Note that Drake does so as a constrained optimization problem
    # decision variables are q
    ik = InverseKinematics(plant, ik_context)
    opt_program = ik.prog()
    q = ik.q()

    world_frame = plant.world_frame()
    end_effector = plant.GetFrameByName("body", wsg_instance)

    # Add Positional Constraints (xyz)
    position_desired = desired_end_effector.translation()
    ik.AddPositionConstraint(
        frameB = end_effector,
        p_BQ=np.zeros(3),
        frameA=world_frame,
        p_AQ_lower=position_desired - position_tol,
        p_AQ_upper=position_desired + position_tol,
    )

    # Add Rotational Constraints
    rotation_desired = desired_end_effector.rotation()
    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=rotation_desired,
        frameBbar=end_effector,
        R_BbarB=RotationMatrix(),
        theta_bound=theta_tol
    )

    # Prefer a solution with the above constraints that is near the current seeded location.
    # This is to prevent multiple solutions for IK that may cause the analytical average to be bad.
    joint_names = [f"iiwa_joint_{i}" for i in range(1, 8)]
    q_iiwa_variables = []
    for joint_name in joint_names:
        joint = plant.GetJointByName(joint_name, iiwa_instance)
        start_pos = joint.position_start()
        number_of_pos = joint.num_positions()
        q_iiwa_variables.extend(list(q[start_pos:start_pos + number_of_pos]))

    q_iiwa_variables = np.array(q_iiwa_variables)

    # Add a stay close convex cost to minimize: ||q_iiwa_seed - q_iiwa_variables||^2
    Q = np.eye(7)
    opt_program.AddQuadraticErrorCost(Q, q_iiwa_seed, q_iiwa_variables)

    # Set initial guess
    opt_program.SetInitialGuess(q, q0_all)
    result = Solve(opt_program)
    if not result.is_success():
        raise Exception("IK Optimization failed")

    # Extract q_goal for the iiwa
    q_solution_all = result.GetSolution(q)

    plant.SetPositions(ik_context, q_solution_all)
    q_goal_iiwa_only = plant.GetPositions(ik_context, iiwa_instance).copy()

    return q_goal_iiwa_only
