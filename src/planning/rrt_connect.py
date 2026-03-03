
import numpy as np

TRAPPED = 0
ADVANCED = 1
REACHED = 2


class Tree:
    def __init__(self, q_root):
        self.nodes = [np.asarray(q_root, dtype=float).copy()]
        self.parent = [-1]

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
    Checks if the sampled edge from q0 to q1 are collision free using L_\inf
    """

    dq = q1 - q0
    dist_inf = np.max(np.abs(dq))
    n = int(np.ceil(dist_inf / resolution))

    # If very close, just check q1
    if n <= 1:
        return bool(is_free(q1))

    for i in range(1, n + 1):
        # start at config 1, skip q0
        a = i / n
        qi = (1 - a) * q0 + a * q1
        if not is_free(qi):
            return False
    return True


def shortcut_path(path, is_free, edge_resolution=0.01, max_passes=None):
    """
    Repeatedly tries to replace path sub-sequences with straight joint-space edges.
    Stops when a full pass finds no valid shortcut, or max_passes is reached.
    """

    path = [np.asarray(q, dtype=float).copy() for q in path]
    if len(path) <= 2:
        return path

    passes = 0
    while True:
        if max_passes is not None and passes >= max_passes:
            break

        improved = False
        n = len(path)

        # Try larger index gaps first to remove as many intermediate nodes as possible.
        for gap in range(n - 1, 1, -1):
            for i in range(0, n - gap):
                j = i + gap
                if edge_is_free(is_free, path[i], path[j], resolution=edge_resolution):
                    path = path[: i + 1] + path[j:]
                    improved = True
                    break
            if improved:
                break

        passes += 1
        if not improved:
            break

    return path

def trace_path(tree: Tree, idx: int):
    """Returns path from root to idx (inclusive)."""
    path = []
    while idx != -1:
        path.append(tree.nodes[idx])
        idx = tree.parent[idx]
    path.reverse()
    return path

def extend_tree(tree: Tree, q_target, is_free, step_size, edge_resolution, joints_lower_limits, joints_upper_limits):
    """
    Try to extend tree by one step toward q_target.
    Returns (status, new_index_or_None).
    """

    q_target = np.asarray(q_target, dtype=float)
    index_near = find_nearest_node(tree.nodes, q_target)
    q_near = tree.nodes[index_near]

    q_new = find_new_node(q_near, q_target, step_size)
    q_new = np.clip(q_new, joints_lower_limits, joints_upper_limits)

    # Edge collision check (includes endpoints)
    if not edge_is_free(is_free, q_near, q_new, resolution=edge_resolution):
        return TRAPPED, None

    tree.nodes.append(q_new)
    tree.parent.append(index_near)
    new_idx = len(tree.nodes) - 1

    # REACHED if we landed exactly on the target
    if np.linalg.norm(q_new - q_target) < 1e-8:
        return REACHED, new_idx

    # otherwise, continue to build tree and return ADVANCED
    return ADVANCED, new_idx

def connect(tree: Tree, q_target, is_free, step_size, edge_resolution, joints_lower_limits, joints_upper_limits, max_connect_steps=10_000):
    """
    Greedily extend tree toward q_target until TRAPPED or REACHED.
    Returns (status, last_index or None if failed).
    """
    last_index = None
    for i in range(max_connect_steps):
        status, index = extend_tree(
            tree, q_target, is_free, step_size, edge_resolution,
            joints_lower_limits, joints_upper_limits
        )

        last_index = index
        if status != ADVANCED:
            return status, last_index

    # if fails
    return TRAPPED, last_index

def rrt_connect_plan(
        q_start,
        q_goal,
        is_free,
        joints_lower_limits,
        joints_upper_limits,
        step_size = 0.1,
        goal_sample_rate = 0.1,
        line_sample_rate=0.55,
        sigma_line=0.20,
        max_iters=30000,
        edge_resolution=0.01,
        max_sample_tries=30,
        max_connect_steps=10000,
        enable_shortcut=True,
        shortcut_edge_resolution=None,
        shortcut_max_passes=None,
):
    """
    Runs RRT-connect from start and goal, then optionally short-cuts the resulting
    waypoint path.
    Returns a list of configurations from q_start to q_goal (inclusive).
    """

    rng = np.random.default_rng()

    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    joints_lower_limits = np.asarray(joints_lower_limits)
    joints_upper_limits = np.asarray(joints_upper_limits)

    T_start = Tree(q_start)
    T_goal = Tree(q_goal)

    # check for feasibility:
    if not is_free(q_start):
        raise RuntimeError("q_start is not a feasible configuration")
    if not is_free(q_goal):
        raise RuntimeError("q_goal is not a feasible configuration")

    # define function for sampling since we call it twice every time for each tree (start & goal)
    def sample():
        """Mixture sampler from original RRT sampler"""
        r = rng.random()

        if r < goal_sample_rate:
            q = q_goal.copy()
        elif r < goal_sample_rate + line_sample_rate:
            # sample bias near the straight-line between start and goal (with added noise)
            u = rng.random()
            q = (1 - u) * q_start + u * q_goal + rng.normal(0.0, sigma_line, size=q_start.shape)
            q = np.clip(q, joints_lower_limits, joints_upper_limits)
        else:
            q = rng.uniform(joints_lower_limits, joints_upper_limits)

        return q

    for iter in range(max_iters):

        # Sample and reject samples that are in collision
        q_rand = None
        for i in range(max_sample_tries):
            cand = sample()
            if is_free(cand):
                q_rand = cand
                break
        if q_rand is None:
            q_rand = sample()

        # Extend active tree one step toward sample
        status_start_branch, idx_start_branch = extend_tree(
            T_start, q_rand, is_free, step_size, edge_resolution,
            joints_lower_limits, joints_upper_limits
        )
        if status_start_branch != TRAPPED:
            q_new = T_start.nodes[idx_start_branch]

            # Greedily connect the other tree toward the new node
            status_goal_branch, idx_goal_branch = connect(
                T_goal, q_new, is_free, step_size, edge_resolution, joints_lower_limits, joints_upper_limits, max_connect_steps=max_connect_steps
            )

            if status_goal_branch == REACHED:
                # We connected at (idx_start_branch in T_start) and (idx_goal_branch in T_goal),
                # and T_goal.nodes[idx_goal_branch] should equal q_new.
                path_from_start = trace_path(T_start, idx_start_branch)
                path_from_goal = trace_path(T_goal, idx_goal_branch)

                # Determine which tree started at q_start
                if np.linalg.norm(T_start.nodes[0] - q_start) < 1e-12:
                    path_start_to_conn = path_from_start
                    path_goal_to_conn = path_from_goal
                else:
                    path_start_to_conn = path_from_goal
                    path_goal_to_conn = path_from_start

                # path_goal_to_conn is root(goal) -> ... -> conn, so reverse it to conn -> ... -> goal
                path = path_start_to_conn + path_goal_to_conn[::-1][1:]
                raw_len = len(path)
                if enable_shortcut:
                    shortcut_resolution = (
                        edge_resolution if shortcut_edge_resolution is None else shortcut_edge_resolution
                    )
                    path = shortcut_path(
                        path,
                        is_free,
                        edge_resolution=shortcut_resolution,
                        max_passes=shortcut_max_passes,
                    )
                print(
                    f"RRT-Connect succeeded in {iter} iterations. "
                    f"Path length: {raw_len} -> {len(path)}"
                )
                return path

        # Swap roles each iteration to advance one another each time
        T_start, T_goal = T_goal, T_start

    raise RuntimeError("RRT-Connect failed.")
