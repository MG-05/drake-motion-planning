import time

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
    max_soft_starts=8,
    soft_start_sigma=0.08,
    soft_start_random_seed=0,
    deadline_s=None,
    log_diagnostics=False,
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
    q_iiwa_seed = np.asarray(q_iiwa_seed, dtype=float).reshape(-1)
    if q_iiwa_seed.shape != (7,):
        raise ValueError(f"q_iiwa_seed must have shape (7,), got {q_iiwa_seed.shape}")

    q_iiwa_vars = []
    iiwa_position_indices = []
    iiwa_lower_limits = []
    iiwa_upper_limits = []
    for name in [f"iiwa_joint_{i}" for i in range(1, 8)]:
        j = plant.GetJointByName(name, iiwa_instance)
        i0 = j.position_start()
        iiwa_position_indices.append(i0)
        q_iiwa_vars.append(q[i0])
        iiwa_lower_limits.append(j.position_lower_limits()[0])
        iiwa_upper_limits.append(j.position_upper_limits()[0])

    iiwa_position_indices = np.asarray(iiwa_position_indices, dtype=int)
    iiwa_lower_limits = np.asarray(iiwa_lower_limits, dtype=float)
    iiwa_upper_limits = np.asarray(iiwa_upper_limits, dtype=float)
    q_iiwa_vars = np.array(q_iiwa_vars)

    prog.AddQuadraticErrorCost(np.eye(7), q_iiwa_seed, q_iiwa_vars)

    # 8) Multi-start initial guesses to reduce IK local-minimum failures.
    max_soft_starts = max(1, int(max_soft_starts))
    q_iiwa_current = q0_all[iiwa_position_indices].copy()
    q_iiwa_center = 0.5 * (iiwa_lower_limits + iiwa_upper_limits)

    soft_start_guesses = []

    def append_soft_start(q_guess_iiwa):
        q_guess_iiwa = np.asarray(q_guess_iiwa, dtype=float).reshape(7)
        q_guess_iiwa = np.clip(q_guess_iiwa, iiwa_lower_limits, iiwa_upper_limits)
        for existing_guess in soft_start_guesses:
            if np.allclose(existing_guess, q_guess_iiwa, atol=1e-8):
                return
        soft_start_guesses.append(q_guess_iiwa)

    # Deterministic first guesses (current/home location)
    append_soft_start(q_iiwa_current)
    append_soft_start(q_iiwa_seed)
    append_soft_start(0.5 * (q_iiwa_current + q_iiwa_seed))
    append_soft_start(q_iiwa_center)
    append_soft_start(2.0 * q_iiwa_center - q_iiwa_seed)

    # Randomized perturbations around seed / center for extra basin coverage.
    rng = np.random.default_rng(soft_start_random_seed)
    joint_ranges = iiwa_upper_limits - iiwa_lower_limits
    sigma = np.maximum(float(soft_start_sigma), 1e-4) * np.maximum(joint_ranges, 1e-3)
    max_random_draws = 10 * max_soft_starts
    random_draws = 0
    while len(soft_start_guesses) < max_soft_starts and random_draws < max_random_draws:
        random_draws += 1
        base = q_iiwa_seed if (random_draws % 2 == 1) else q_iiwa_center
        append_soft_start(base + rng.normal(0.0, sigma, size=7))

    soft_start_guesses = soft_start_guesses[:max_soft_starts]

    # Solve from each soft start, return immediately on first success.
    for attempt_idx, q_guess_iiwa in enumerate(soft_start_guesses, start=1):
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            raise TimeoutError("IK exceeded its planning deadline")
        q_guess_all = q0_all.copy()
        q_guess_all[iiwa_position_indices] = q_guess_iiwa
        prog.SetInitialGuess(q, q_guess_all)
        result = Solve(prog)
        if result.is_success():
            q_sol = result.GetSolution(q)
            plant.SetPositions(plant_context, q_sol)
            if bool(log_diagnostics):
                print(f"IK succeeded on soft-start attempt {attempt_idx}/{len(soft_start_guesses)}")
            return plant.GetPositions(plant_context, iiwa_instance).copy()

    if bool(log_diagnostics):
        print(f"IK failed after {len(soft_start_guesses)} soft-start attempts")
        print("Solver:", result.get_solver_id().name())
        print("SolutionResult:", result.get_solution_result())
        try:
            print("InfeasibleConstraintNames:", result.GetInfeasibleConstraintNames(prog))
        except Exception as e:
            print("Infeasible-constraint reporting unavailable:", e)
    raise RuntimeError("IK Optimization failed across all soft starts")
