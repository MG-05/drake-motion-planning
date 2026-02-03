import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.math import RotationMatrix
from pydrake.solvers import Solve


def solve_iiwa_ik_for_gripper_pose(
    plant,
    root_context_current,
    iiwa_instance,
    wsg_instance,
    desired_end_effector,
    q_iiwa_seed,
    position_tol=0.01,
    theta_tol=0.01,
):
    # 1) Clone the ROOT context (so we don't mutate simulator state and because it errors otherwise)
    ik_root_context = root_context_current.Clone()

    # 2) Extract the PLANT subcontext from the cloned root
    plant_context = plant.GetMyMutableContextFromRoot(ik_root_context)

    # 3) Read the current/start generalized positions
    q0_all = plant.GetPositions(plant_context).copy()

    # 4) Build IK on the PLANT context
    ik = InverseKinematics(plant, plant_context)
    prog = ik.prog()
    q = ik.q()

    # Fix wsg fingers
    for finger_name in ["left_finger_sliding_joint", "right_finger_sliding_joint"]:
        joint = plant.GetJointByName(finger_name, wsg_instance)
        i0 = joint.position_start()
        n = joint.num_positions()
        # iterate over all joints that belong to the finger model and retrieve each joint object
        for k in range(n):
            # put a bounding box for each joint that occupies indices i0 to i0+n-1
            idx = i0 + k
            prog.AddBoundingBoxConstraint(q0_all[idx], q0_all[idx], q[idx])

    # freeze brick when available:
    if plant.HasModelInstanceNamed("foam_brick"):
        brick = plant.GetModelInstanceByName("foam_brick")
        for joint_index in plant.GetJointIndices(brick):
            joint = plant.get_joint(joint_index)
            i0 = joint.position_start()
            n = joint.num_positions()
            # iterate over all joints that belong to the finger model and retrieve each joint objec
            for k in range(n):
                # put a bounding box for each joint that occupies indices i0 to i0+n-1
                idx = i0 + k
                prog.AddBoundingBoxConstraint(q0_all[idx], q0_all[idx], q[idx])

    world_frame = plant.world_frame()
    end_effector = plant.GetFrameByName("body", wsg_instance)

    # 5) Position constraint
    p_W_des = desired_end_effector.translation()
    ik.AddPositionConstraint(
        frameB=end_effector,
        p_BQ=np.zeros(3),
        frameA=world_frame,
        p_AQ_lower=p_W_des - position_tol,
        p_AQ_upper=p_W_des + position_tol,
    )

    # 6) Orientation constraint
    R_W_des = desired_end_effector.rotation()
    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=R_W_des,
        frameBbar=end_effector,
        R_BbarB=RotationMatrix(),
        theta_bound=theta_tol,
    )

    # 7) Add cost to stay near seed (inital q0) on iiwa joints
    q_iiwa_vars = []
    for name in [f"iiwa_joint_{i}" for i in range(1, 8)]:
        j = plant.GetJointByName(name, iiwa_instance)
        i0 = j.position_start()
        q_iiwa_vars.append(q[i0])
    q_iiwa_vars = np.array(q_iiwa_vars)

    prog.AddQuadraticErrorCost(np.eye(7), q_iiwa_seed, q_iiwa_vars)

    # 8) Initial guess and solve
    prog.SetInitialGuess(q, q0_all)
    result = Solve(prog)
    # info if IK solver fails
    if not result.is_success():
        print("Solver:", result.get_solver_id().name())
        print("SolutionResult:", result.get_solution_result())
        try:
            print("InfeasibleConstraintNames:", result.GetInfeasibleConstraintNames(prog))
        except Exception as e:
            print("Infeasible-constraint reporting unavailable:", e)
        raise RuntimeError("IK Optimization failed")

    q_sol = result.GetSolution(q)

    # 9) Extract iiwa-only solution using the PLANT context
    plant.SetPositions(plant_context, q_sol)
    return plant.GetPositions(plant_context, iiwa_instance).copy()
