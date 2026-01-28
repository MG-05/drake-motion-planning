import numpy as np

def find_nearest_node(nodes, q):
    """
    Returns the index of the closest node in nodes to point q, the target sample.
    nodes is a list of configs (7D b/c of 7 joints in iiwa arm) in the RRT tree
    q is the 7D target point
    """

    d = [np.linalg.norm(n - q) for n in nodes]
    return int(np.argmin(d))

def find_new_node(q_origin, q_dest, step_size):
    """
    Returns a new configuration that is at most step_size away from the origin. The
    new node is aimed at q_dest.
    """
    v = q_dest - q_origin
    dist = np.linalg.norm(v)
    if dist <= step_size:
        return q_dest.copy()

    return q_origin + (step_size/dist) * v

def edge_is_free(is_free, q0, q1, resolution=0.02):
    """
    Checks if the sampled edge from q0 to q1 are collision free.
    """

    dist = np.linalg.norm(q1 - q0)
    n = int(np.ceil(dist / resolution)) + 1

    for a in np.linspace(0, 1, n):
        connecting_line = (1-a)*q0 + a*q1
        if not is_free(connecting_line):
            return False

    return True

def rrt_plan(
        q_start,
        q_goal,
        is_free,
        joints_lower_limits,
        joints_upper_limits,
        step_size = 0.1,
        goal_sample_rate = 0.1,
        max_iters = 5000,
        edge_resolution=0.02,
        goal_tolerance=0.1,
):
    """
    runs RRT sampler from q_start to q_goal, returning a list of configurations (path)
    note that we have a goal_sample_rate, so we can sample a point towards the goal on a
    probability so we can diversify and sample outwards in the case of a maze (prevents greedy
    traps)
    """

    rng = np.random.default_rng()

    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    joints_lower_limits = np.asarray(joints_lower_limits)
    joints_upper_limits = np.asarray(joints_upper_limits)

    # check for feasibility:
    if not is_free(q_start):
        raise RuntimeError("q_start is not a feasible configuration")
    if not is_free(q_goal):
        raise RuntimeError("q_goal is not a feasible configuration")

    nodes = [q_start.copy()]
    parent = [-1]

    for i in range(max_iters):
        # if rng.random() < goal_sample_rate:
        #     q_random = q_goal
        # else:
        #     q_random = rng.uniform(joints_lower_limits, joints_upper_limits)

        r = rng.random()
        if r < goal_sample_rate:
            q_random = q_goal
        elif r < goal_sample_rate + 0.65:
            # sample near the straight-line between start and goal (with added noise)
            u = rng.random()
            # in radians
            sigma = 0.25
            q_random = (1 - u) * q_start + u * q_goal + rng.normal(0.0, sigma, size=q_start.shape)
            q_random = np.clip(q_random, joints_lower_limits, joints_upper_limits)
        else:
            q_random = rng.uniform(joints_lower_limits, joints_upper_limits)

        i_near = find_nearest_node(nodes, q_random)
        q_new = find_new_node(nodes[i_near], q_random, step_size)

        # skip if sampled edge is in collision
        if not edge_is_free(is_free, nodes[i_near], q_new, resolution=edge_resolution):
            continue

        # add to RRT tree
        nodes.append(q_new)
        parent.append(i_near)

        if np.linalg.norm(q_new - q_goal) < goal_tolerance:
#             connect directly to goal if within the goal_tolerance
            if edge_is_free(is_free, q_new, q_goal, resolution=edge_resolution):
                nodes.append(q_goal.copy())
                parent.append(len(nodes) - 2)
                print(f"RRT succeeded in {i} iterations")
                break

    # Reconstruct if goal reached
    if np.linalg.norm(nodes[-1] - q_goal) > 1e-9:
        raise RuntimeError("RRT failed to reach goal (increase max_iters, adjust step_size, or change goal).")

    # build path from q_goal to q_start
    path = []
    index = len(nodes) - 1
    while index >= 0:
        path.append(nodes[index])
        index = parent[index]

    path.reverse()
    return path
