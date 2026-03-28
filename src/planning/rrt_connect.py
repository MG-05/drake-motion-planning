import dataclasses as dc
import numpy as np
import time

TRAPPED = 0
ADVANCED = 1
REACHED = 2


@dc.dataclass(frozen=True)
class AdaptiveStepConfig:
    """
    Online step-size adaptation for RRT-Connect tree expansion.

    `min_step_size` and `max_step_size` are in joint-space radians.
    `clearance_gain` maps clearance margin in meters to an additional joint-space
    step budget. The result is clipped into `[min_step_size, max_step_size]`.

    The persistent per-tree step scale is controlled by a smoothed success-rate
    controller with Schmitt-trigger hysteresis:
    - the tree enters "shrink" mode when the EMA falls below
      `success_rate_lower_trigger`
    - the tree enters "grow" mode when the EMA rises above
      `success_rate_upper_trigger`
    - between the two thresholds it keeps its previous mode

    Scale updates are then applied only when the latest extension outcome agrees
    with the active mode, which avoids the twitchiness of raw per-extension
    success/failure scaling.
    """
    enabled: bool = True
    min_step_size: float = 0.05
    max_step_size: float | None = None
    clearance_gain: float = 15.0
    max_backoff_trials: int = 3
    backoff_factor: float = 0.5
    success_rate_ema_alpha: float = 0.15
    success_rate_lower_trigger: float = 0.35
    success_rate_upper_trigger: float = 0.65
    step_scale_growth_factor: float = 1.08
    step_scale_shrink_factor: float = 0.88
    initial_success_rate: float = 0.50
    tree_scale_min: float = 0.5
    tree_scale_max: float = 2.0


@dc.dataclass(frozen=True)
class RRTConnectConfig:
    """
    Shared configuration for all RRT-Connect calls in the manipulation stack.

    The adaptive controller is the mechanism for handling clutter vs open space
    online, so this profile should be reused across phases instead of tuning
    separate stage-specific step sizes.
    """
    step_size: float = 0.7
    goal_sample_rate: float = 0.25
    line_sample_rate: float = 0.55
    sigma_line: float = 0.20
    max_iters: int = 50_000
    edge_resolution: float = 0.03
    max_sample_tries: int = 30
    max_connect_steps: int = 1000
    enable_shortcut: bool = True
    shortcut_edge_resolution: float | None = None
    shortcut_max_passes: int | None = None
    adaptive_step_config: AdaptiveStepConfig = dc.field(
        default_factory=lambda: AdaptiveStepConfig(
            enabled=True,
            min_step_size=0.03,
            max_step_size=1.0,
            clearance_gain=20.0,
            max_backoff_trials=1,
            backoff_factor=0.50,
            success_rate_ema_alpha=0.13,
            success_rate_lower_trigger=0.35,
            success_rate_upper_trigger=0.70,
            step_scale_growth_factor=1.13,
            step_scale_shrink_factor=0.93,
            initial_success_rate=0.50,
            tree_scale_min=0.5,
            tree_scale_max=3.0,
        )
    )

    def to_plan_kwargs(self, *, deadline_s: float | None = None) -> dict[str, object]:
        return {
            "step_size": float(self.step_size),
            "goal_sample_rate": float(self.goal_sample_rate),
            "line_sample_rate": float(self.line_sample_rate),
            "sigma_line": float(self.sigma_line),
            "max_iters": int(self.max_iters),
            "edge_resolution": float(self.edge_resolution),
            "max_sample_tries": int(self.max_sample_tries),
            "max_connect_steps": int(self.max_connect_steps),
            "enable_shortcut": bool(self.enable_shortcut),
            "shortcut_edge_resolution": self.shortcut_edge_resolution,
            "shortcut_max_passes": self.shortcut_max_passes,
            "deadline_s": deadline_s,
            "adaptive_step_config": self.adaptive_step_config,
        }


class Tree:
    def __init__(self, q_root, adaptive_step_config: AdaptiveStepConfig | None = None):
        self.nodes = [np.asarray(q_root, dtype=float).copy()]
        self.parent = [-1]
        self.step_scale = 1.0
        self.node_clearance = [None]
        if adaptive_step_config is None or not adaptive_step_config.enabled:
            self.success_rate_ema = 0.5
        else:
            self.success_rate_ema = float(
                np.clip(adaptive_step_config.initial_success_rate, 0.0, 1.0)
            )
        self.step_scale_control_mode: str | None = None

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

def _clip_step_scale(tree: Tree, config: AdaptiveStepConfig):
    tree.step_scale = float(
        np.clip(tree.step_scale, config.tree_scale_min, config.tree_scale_max)
    )

def _validate_adaptive_step_config(config: AdaptiveStepConfig | None):
    if config is None or not config.enabled:
        return

    if not (0.0 < float(config.success_rate_ema_alpha) <= 1.0):
        raise ValueError("success_rate_ema_alpha must lie in (0, 1].")
    if not (0.0 <= float(config.initial_success_rate) <= 1.0):
        raise ValueError("initial_success_rate must lie in [0, 1].")

    lower = float(config.success_rate_lower_trigger)
    upper = float(config.success_rate_upper_trigger)
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(
            "success_rate_lower_trigger and success_rate_upper_trigger must satisfy "
            "0 <= lower < upper <= 1."
        )
    if float(config.step_scale_growth_factor) < 1.0:
        raise ValueError("step_scale_growth_factor must be at least 1.0.")
    if not (0.0 < float(config.step_scale_shrink_factor) <= 1.0):
        raise ValueError("step_scale_shrink_factor must lie in (0, 1].")

def _resolve_step_bounds(step_size, adaptive_step_config: AdaptiveStepConfig | None):
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        nominal_step_size = float(step_size)
        return nominal_step_size, nominal_step_size

    min_step_size = max(1e-6, float(adaptive_step_config.min_step_size))
    max_step_size = (
        float(step_size)
        if adaptive_step_config.max_step_size is None else float(adaptive_step_config.max_step_size)
    )
    if max_step_size < min_step_size:
        max_step_size = min_step_size
    return min_step_size, max_step_size

def _lookup_clearance_estimate(
    tree: Tree,
    node_index: int,
    q_node,
    is_free,
):
    cached_clearance = tree.node_clearance[node_index]
    if cached_clearance is not None:
        return float(cached_clearance)

    estimate_clearance = getattr(is_free, "estimate_clearance", None)
    if not callable(estimate_clearance):
        return None

    clearance = float(estimate_clearance(q_node))
    if np.isfinite(clearance):
        tree.node_clearance[node_index] = clearance
        return clearance
    return None

def _update_step_scale_controller(
    tree: Tree,
    adaptive_step_config: AdaptiveStepConfig | None,
    extension_succeeded: bool,
):
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return

    alpha = float(adaptive_step_config.success_rate_ema_alpha)
    outcome = 1.0 if extension_succeeded else 0.0
    tree.success_rate_ema = (
        (1.0 - alpha) * float(tree.success_rate_ema)
        + alpha * outcome
    )

    lower = float(adaptive_step_config.success_rate_lower_trigger)
    upper = float(adaptive_step_config.success_rate_upper_trigger)
    if tree.success_rate_ema <= lower:
        tree.step_scale_control_mode = "shrink"
    elif tree.success_rate_ema >= upper:
        tree.step_scale_control_mode = "grow"

    if tree.step_scale_control_mode == "grow" and extension_succeeded:
        tree.step_scale *= float(adaptive_step_config.step_scale_growth_factor)
    elif tree.step_scale_control_mode == "shrink" and not extension_succeeded:
        tree.step_scale *= float(adaptive_step_config.step_scale_shrink_factor)
    else:
        return

    _clip_step_scale(tree, adaptive_step_config)

def _compute_adaptive_step_size(
    tree: Tree,
    q_near,
    node_index,
    is_free,
    step_size,
    adaptive_step_config: AdaptiveStepConfig | None,
    use_clearance_estimate: bool,
):
    min_step_size, max_step_size = _resolve_step_bounds(step_size, adaptive_step_config)
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return max_step_size

    local_step_size = max_step_size
    if use_clearance_estimate:
        clearance = _lookup_clearance_estimate(tree, node_index, q_near, is_free)
        if clearance is not None:
            min_clearance = float(getattr(is_free, "minimum_clearance", 0.0))
            clearance_margin = max(0.0, clearance - min_clearance)
            local_step_size = min_step_size + float(adaptive_step_config.clearance_gain) * clearance_margin

    local_step_size *= float(tree.step_scale)
    return float(np.clip(local_step_size, min_step_size, max_step_size))

def extend_tree(
    tree: Tree,
    q_target,
    is_free,
    step_size,
    edge_resolution,
    joints_lower_limits,
    joints_upper_limits,
    adaptive_step_config: AdaptiveStepConfig | None = None,
):
    """
    Try to extend tree by one step toward q_target.
    Returns (status, new_index_or_None).
    """

    q_target = np.asarray(q_target, dtype=float)
    index_near = find_nearest_node(tree.nodes, q_target)
    q_near = tree.nodes[index_near]

    candidate_step_size = _compute_adaptive_step_size(
        tree,
        q_near,
        index_near,
        is_free,
        step_size,
        adaptive_step_config,
        use_clearance_estimate=(tree.node_clearance[index_near] is not None),
    )
    attempt_count = 1
    if adaptive_step_config is not None and adaptive_step_config.enabled:
        attempt_count = max(1, int(adaptive_step_config.max_backoff_trials) + 1)

    q_new = None
    for attempt_idx in range(attempt_count):
        q_new = find_new_node(q_near, q_target, candidate_step_size)
        q_new = np.clip(q_new, joints_lower_limits, joints_upper_limits)

        if np.linalg.norm(q_new - q_near) <= 1e-12:
            break
        if edge_is_free(is_free, q_near, q_new, resolution=edge_resolution):
            tree.nodes.append(q_new)
            tree.parent.append(index_near)
            tree.node_clearance.append(None)
            new_idx = len(tree.nodes) - 1

            _update_step_scale_controller(
                tree,
                adaptive_step_config,
                extension_succeeded=True,
            )

            # REACHED if we landed exactly on the target
            if np.linalg.norm(q_new - q_target) < 1e-8:
                return REACHED, new_idx
            return ADVANCED, new_idx

        if adaptive_step_config is None or not adaptive_step_config.enabled:
            break
        # Query clearance only after a failed aggressive attempt, then reuse it.
        if attempt_idx == 0 and tree.node_clearance[index_near] is None:
            clearance_step_size = _compute_adaptive_step_size(
                tree,
                q_near,
                index_near,
                is_free,
                step_size,
                adaptive_step_config,
                use_clearance_estimate=True,
            )
            candidate_step_size = min(candidate_step_size, clearance_step_size)
        candidate_step_size *= float(adaptive_step_config.backoff_factor)
        min_step_size, _ = _resolve_step_bounds(step_size, adaptive_step_config)
        if candidate_step_size < min_step_size:
            candidate_step_size = min_step_size

    _update_step_scale_controller(
        tree,
        adaptive_step_config,
        extension_succeeded=False,
    )

    return TRAPPED, None

def connect(
    tree: Tree,
    q_target,
    is_free,
    step_size,
    edge_resolution,
    joints_lower_limits,
    joints_upper_limits,
    max_connect_steps=10_000,
    adaptive_step_config: AdaptiveStepConfig | None = None,
):
    """
    Greedily extend tree toward q_target until TRAPPED or REACHED.
    Returns (status, last_index or None if failed).
    """
    last_index = None
    for i in range(max_connect_steps):
        status, index = extend_tree(
            tree, q_target, is_free, step_size, edge_resolution,
            joints_lower_limits, joints_upper_limits,
            adaptive_step_config=adaptive_step_config,
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
        deadline_s=None,
        adaptive_step_config: AdaptiveStepConfig | None = None,
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

    _validate_adaptive_step_config(adaptive_step_config)

    T_start = Tree(q_start, adaptive_step_config=adaptive_step_config)
    T_goal = Tree(q_goal, adaptive_step_config=adaptive_step_config)

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
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            raise TimeoutError("RRT-Connect timed out before finding a path.")

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
            joints_lower_limits, joints_upper_limits,
            adaptive_step_config=adaptive_step_config,
        )
        if status_start_branch != TRAPPED:
            q_new = T_start.nodes[idx_start_branch]

            # Greedily connect the other tree toward the new node
            status_goal_branch, idx_goal_branch = connect(
                T_goal, q_new, is_free, step_size, edge_resolution,
                joints_lower_limits, joints_upper_limits,
                max_connect_steps=max_connect_steps,
                adaptive_step_config=adaptive_step_config,
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
