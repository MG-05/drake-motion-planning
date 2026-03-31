import dataclasses as dc
import numpy as np
import time

TRAPPED = 0
ADVANCED = 1
REACHED = 2


@dc.dataclass(frozen=True)
class AdaptiveStepConfig:
    """
    Node-local adaptive control for RRT-Connect expansion.

    The planner maintains a preferred step scale per node, shrinks that scale
    immediately on failed extensions, and updates it toward the step size that
    actually succeeded after any local backoff. Nodes that repeatedly fail also
    attract fewer distant samples through a dynamic-domain style sampling
    radius. Optional clearance queries are only triggered on trapped nodes and
    cached as local step caps.
    """
    enabled: bool = True
    min_step_size: float = 0.05
    max_step_size: float | None = None
    max_backoff_trials: int = 3
    backoff_factor: float = 0.65
    successful_step_ema_alpha: float = 0.35
    step_scale_growth_factor: float = 1.05
    step_scale_shrink_factor: float = 0.70
    initial_node_scale: float = 0.80
    node_scale_min: float = 0.20
    node_scale_max: float = 1.25
    min_connect_steps: int = 32
    sample_domain_step_multiplier: float = 3.0
    sample_domain_initial_scale: float = 1.0
    sample_domain_growth_factor: float = 1.05
    sample_domain_shrink_factor: float = 0.60
    sample_domain_scale_min: float = 0.35
    sample_domain_scale_max: float = 2.0
    failure_node_sample_rate: float = 0.15
    failure_node_sample_sigma_scale: float = 0.75
    clearance_step_cap_gain: float = 18.0


@dc.dataclass(frozen=True)
class RRTConnectConfig:
    """
    Shared configuration for all RRT-Connect calls in the manipulation stack.

    The adaptive controller is the mechanism for handling clutter vs open space
    online, so this profile should be reused across phases instead of tuning
    separate stage-specific step sizes. Shortcutting is configured here as an
    optional post-process, but it is not part of the search itself.
    """
    step_size: float = 0.9
    goal_sample_rate: float = 0.30
    line_sample_rate: float = 0.20
    sigma_line: float = 0.18
    max_iters: int = 50_000
    edge_resolution: float = 0.07
    max_sample_tries: int = 30
    max_connect_steps: int = 250
    enable_shortcut: bool = True
    shortcut_edge_resolution: float | None = None
    shortcut_max_attempts: int | None = 128
    shortcut_time_budget_s: float | None = 1.0
    final_validation_edge_resolution: float | None = 0.03
    random_seed: int | None = 0
    adaptive_step_config: AdaptiveStepConfig = dc.field(
        default_factory=lambda: AdaptiveStepConfig(
            enabled=True,
            min_step_size=0.04,
            max_step_size=1.0,
            max_backoff_trials=3,
            backoff_factor=0.65,
            successful_step_ema_alpha=0.35,
            step_scale_growth_factor=1.05,
            step_scale_shrink_factor=0.70,
            initial_node_scale=0.85,
            node_scale_min=0.20,
            node_scale_max=1.25,
            min_connect_steps=24,
            sample_domain_step_multiplier=3.0,
            sample_domain_initial_scale=1.0,
            sample_domain_growth_factor=1.05,
            sample_domain_shrink_factor=0.60,
            sample_domain_scale_min=0.35,
            sample_domain_scale_max=2.0,
            failure_node_sample_rate=0.15,
            failure_node_sample_sigma_scale=0.75,
            clearance_step_cap_gain=18.0,
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
            "final_validation_edge_resolution": self.final_validation_edge_resolution,
            "deadline_s": deadline_s,
            "random_seed": self.random_seed,
            "adaptive_step_config": self.adaptive_step_config,
        }


@dc.dataclass
class RRTConnectTelemetry:
    feasible_goals: int = 0
    iterations: int = 0
    direct_edge_checks: int = 0
    direct_edge_growth_successes: int = 0
    direct_edge_validation_successes: int = 0
    direct_edge_validation_failures: int = 0
    sample_attempts: int = 0
    sample_rejections: int = 0
    sample_successes: int = 0
    extend_trapped: int = 0
    extend_advanced: int = 0
    connect_trapped: int = 0
    connect_advanced: int = 0
    connect_reached: int = 0
    candidate_paths_found: int = 0
    candidate_paths_rejected_final_validation: int = 0
    elapsed_s: float = 0.0

    def finalize(self, search_start_s: float) -> None:
        self.elapsed_s = float(time.perf_counter() - float(search_start_s))

    def summary(self) -> str:
        return (
            "RRT telemetry: "
            f"elapsed_s={self.elapsed_s:.3f}, "
            f"feasible_goals={self.feasible_goals}, "
            f"iterations={self.iterations}, "
            f"direct_edge_checks={self.direct_edge_checks}, "
            f"direct_edge_growth_successes={self.direct_edge_growth_successes}, "
            f"direct_edge_validation_successes={self.direct_edge_validation_successes}, "
            f"direct_edge_validation_failures={self.direct_edge_validation_failures}, "
            f"sample_attempts={self.sample_attempts}, "
            f"sample_rejections={self.sample_rejections}, "
            f"sample_successes={self.sample_successes}, "
            f"extend_trapped={self.extend_trapped}, "
            f"extend_advanced={self.extend_advanced}, "
            f"connect_trapped={self.connect_trapped}, "
            f"connect_advanced={self.connect_advanced}, "
            f"connect_reached={self.connect_reached}, "
            f"candidate_paths_found={self.candidate_paths_found}, "
            f"candidate_paths_rejected_final_validation={self.candidate_paths_rejected_final_validation}"
        )


class Tree:
    def __init__(self, q_root, adaptive_step_config: AdaptiveStepConfig | None = None):
        self.nodes = [np.asarray(q_root, dtype=float).copy()]
        self.parent = [-1]
        initial_scale = 1.0
        initial_sample_domain_scale = 1.0
        if adaptive_step_config is not None and adaptive_step_config.enabled:
            initial_scale = float(adaptive_step_config.initial_node_scale)
            initial_scale = float(
                np.clip(
                    initial_scale,
                    adaptive_step_config.node_scale_min,
                    adaptive_step_config.node_scale_max,
                )
            )
            initial_sample_domain_scale = float(
                np.clip(
                    adaptive_step_config.sample_domain_initial_scale,
                    adaptive_step_config.sample_domain_scale_min,
                    adaptive_step_config.sample_domain_scale_max,
                )
            )
        self.node_step_scales = [initial_scale]
        self.node_sample_domain_scales = [initial_sample_domain_scale]
        self.node_failure_counts = [0]
        self.node_step_caps = [None]

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

def edge_is_free(is_free, q0, q1, resolution=0.02, deadline_s: float | None = None):
    """
    Checks if the sampled edge from q0 to q1 are collision free using L_inf
    """

    dq = q1 - q0
    dist_inf = np.max(np.abs(dq))
    n = int(np.ceil(dist_inf / resolution))

    # If very close, just check q1
    if n <= 1:
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            raise TimeoutError("RRT-Connect timed out before finding a path.")
        return bool(is_free(q1))

    for i in range(1, n + 1):
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            raise TimeoutError("RRT-Connect timed out before finding a path.")
        # start at config 1, skip q0
        a = i / n
        qi = (1 - a) * q0 + a * q1
        if not is_free(qi):
            return False
    return True


def _copy_joint_path(path) -> list[np.ndarray]:
    return [np.asarray(q, dtype=float).copy() for q in path]


def _joint_path_length(path) -> float:
    path = _copy_joint_path(path)
    if len(path) <= 1:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def randomized_shortcut_path(
    path,
    is_free,
    *,
    edge_resolution=0.01,
    max_attempts: int | None = None,
    random_seed: int | None = None,
    deadline_s: float | None = None,
):
    """
    Randomized shortcutting over waypoint indices with a hard attempt budget.

    This is intentionally a cheap proposal stage. Callers are expected to run
    one strict final validation pass before accepting the shortened result.
    """

    path = _copy_joint_path(path)
    if len(path) <= 2:
        return path

    rng = np.random.default_rng(random_seed)
    attempts = 0
    while len(path) > 2:
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            raise TimeoutError("RRT-Connect timed out before finding a path.")
        if max_attempts is not None and attempts >= int(max_attempts):
            break

        attempts += 1
        n = len(path)
        i = int(rng.integers(0, n - 2))
        j = int(rng.integers(i + 2, n))
        if edge_is_free(
            is_free,
            path[i],
            path[j],
            resolution=edge_resolution,
            deadline_s=deadline_s,
        ):
            path = path[: i + 1] + path[j:]

    return path


def _resolve_shortcut_deadline(
    shortcut_time_budget_s: float | None,
    planning_deadline_s: float | None,
) -> float | None:
    shortcut_deadline_s = planning_deadline_s
    if shortcut_time_budget_s is not None:
        local_deadline_s = time.perf_counter() + max(0.0, float(shortcut_time_budget_s))
        if shortcut_deadline_s is None:
            shortcut_deadline_s = local_deadline_s
        else:
            shortcut_deadline_s = min(shortcut_deadline_s, local_deadline_s)
    return shortcut_deadline_s


def postprocess_rrt_path(
    path,
    is_free,
    *,
    planner_config: RRTConnectConfig,
    deadline_s: float | None = None,
):
    """
    Optionally smooths an already-valid RRT path while preserving the raw path as
    fallback if shortcutting does not help, times out, or fails strict validation.
    """

    raw_path = _copy_joint_path(path)
    if len(raw_path) <= 2 or not bool(planner_config.enable_shortcut):
        return raw_path

    proposal_checker = getattr(is_free, "growth_is_free", is_free)
    strict_checker = getattr(proposal_checker, "strict_is_free", is_free)
    shortcut_resolution = (
        float(planner_config.edge_resolution)
        if planner_config.shortcut_edge_resolution is None
        else float(planner_config.shortcut_edge_resolution)
    )
    final_validation_edge_resolution = _resolve_validation_resolution(
        planner_config.edge_resolution,
        planner_config.final_validation_edge_resolution,
    )
    shortcut_deadline_s = _resolve_shortcut_deadline(
        planner_config.shortcut_time_budget_s,
        deadline_s,
    )

    try:
        candidate_path = randomized_shortcut_path(
            raw_path,
            proposal_checker,
            edge_resolution=shortcut_resolution,
            max_attempts=planner_config.shortcut_max_attempts,
            random_seed=planner_config.random_seed,
            deadline_s=shortcut_deadline_s,
        )
    except TimeoutError:
        return raw_path

    if _joint_path_length(candidate_path) >= _joint_path_length(raw_path) - 1e-12:
        return raw_path

    try:
        if not _path_is_dense_free(
            candidate_path,
            strict_checker,
            edge_resolution=final_validation_edge_resolution,
            deadline_s=shortcut_deadline_s,
        ):
            return raw_path
    except TimeoutError:
        return raw_path

    return candidate_path

def trace_path(tree: Tree, idx: int):
    """Returns path from root to idx (inclusive)."""
    path = []
    while idx != -1:
        path.append(tree.nodes[idx])
        idx = tree.parent[idx]
    path.reverse()
    return path

def trace_path_with_root_index(tree: Tree, idx: int):
    """Returns (path from root to idx, root_index)."""
    path = []
    root_index = idx
    while idx != -1:
        root_index = idx
        path.append(tree.nodes[idx])
        idx = tree.parent[idx]
    path.reverse()
    return path, root_index

def append_tree_root(
    tree: Tree,
    q_root,
):
    tree.nodes.append(np.asarray(q_root, dtype=float).copy())
    tree.parent.append(-1)
    tree.node_step_scales.append(float(tree.node_step_scales[0]))
    tree.node_sample_domain_scales.append(float(tree.node_sample_domain_scales[0]))
    tree.node_failure_counts.append(0)
    tree.node_step_caps.append(None)

def _clip_node_step_scale(scale: float, config: AdaptiveStepConfig) -> float:
    return float(np.clip(scale, config.node_scale_min, config.node_scale_max))

def _clip_sample_domain_scale(scale: float, config: AdaptiveStepConfig) -> float:
    return float(np.clip(scale, config.sample_domain_scale_min, config.sample_domain_scale_max))

def _validate_adaptive_step_config(config: AdaptiveStepConfig | None):
    if config is None or not config.enabled:
        return

    if int(config.max_backoff_trials) < 0:
        raise ValueError("max_backoff_trials must be nonnegative.")
    if not (0.0 < float(config.backoff_factor) <= 1.0):
        raise ValueError("backoff_factor must lie in (0, 1].")
    if not (0.0 < float(config.successful_step_ema_alpha) <= 1.0):
        raise ValueError("successful_step_ema_alpha must lie in (0, 1].")
    if float(config.step_scale_growth_factor) < 1.0:
        raise ValueError("step_scale_growth_factor must be at least 1.0.")
    if not (0.0 < float(config.step_scale_shrink_factor) <= 1.0):
        raise ValueError("step_scale_shrink_factor must lie in (0, 1].")
    if float(config.node_scale_min) <= 0.0:
        raise ValueError("node_scale_min must be positive.")
    if float(config.node_scale_max) < float(config.node_scale_min):
        raise ValueError("node_scale_max must be at least node_scale_min.")
    initial_node_scale = float(config.initial_node_scale)
    if not (float(config.node_scale_min) <= initial_node_scale <= float(config.node_scale_max)):
        raise ValueError(
            "initial_node_scale must lie within [node_scale_min, node_scale_max]."
        )
    if int(config.min_connect_steps) <= 0:
        raise ValueError("min_connect_steps must be positive.")
    if float(config.sample_domain_step_multiplier) <= 0.0:
        raise ValueError("sample_domain_step_multiplier must be positive.")
    if float(config.sample_domain_growth_factor) < 1.0:
        raise ValueError("sample_domain_growth_factor must be at least 1.0.")
    if not (0.0 < float(config.sample_domain_shrink_factor) <= 1.0):
        raise ValueError("sample_domain_shrink_factor must lie in (0, 1].")
    if float(config.sample_domain_scale_min) <= 0.0:
        raise ValueError("sample_domain_scale_min must be positive.")
    if float(config.sample_domain_scale_max) < float(config.sample_domain_scale_min):
        raise ValueError(
            "sample_domain_scale_max must be at least sample_domain_scale_min."
        )
    initial_sample_domain_scale = float(config.sample_domain_initial_scale)
    if not (
        float(config.sample_domain_scale_min)
        <= initial_sample_domain_scale
        <= float(config.sample_domain_scale_max)
    ):
        raise ValueError(
            "sample_domain_initial_scale must lie within "
            "[sample_domain_scale_min, sample_domain_scale_max]."
        )
    if not (0.0 <= float(config.failure_node_sample_rate) <= 1.0):
        raise ValueError("failure_node_sample_rate must lie in [0, 1].")
    if float(config.failure_node_sample_sigma_scale) <= 0.0:
        raise ValueError("failure_node_sample_sigma_scale must be positive.")
    if float(config.clearance_step_cap_gain) < 0.0:
        raise ValueError("clearance_step_cap_gain must be nonnegative.")

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

def _resolve_validation_resolution(
    edge_resolution: float,
    final_validation_edge_resolution: float | None,
) -> float:
    edge_resolution = max(1e-6, float(edge_resolution))
    if final_validation_edge_resolution is None:
        return edge_resolution
    return min(edge_resolution, max(1e-6, float(final_validation_edge_resolution)))

def _compute_node_step_size(
    tree: Tree,
    node_index: int,
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
):
    min_step_size, max_step_size = _resolve_step_bounds(step_size, adaptive_step_config)
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return max_step_size

    node_scale = float(tree.node_step_scales[node_index])
    local_step_size = float(step_size) * node_scale
    local_step_cap = tree.node_step_caps[node_index]
    if local_step_cap is not None:
        local_step_size = min(local_step_size, float(local_step_cap))
    return float(np.clip(local_step_size, min_step_size, max_step_size))

def _compute_node_sample_domain_radius(
    tree: Tree,
    node_index: int,
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
):
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return float("inf")

    local_step_size = _compute_node_step_size(
        tree,
        node_index,
        step_size,
        adaptive_step_config,
    )
    return max(
        local_step_size,
        float(adaptive_step_config.sample_domain_step_multiplier)
        * local_step_size
        * float(tree.node_sample_domain_scales[node_index]),
    )

def _maybe_update_node_step_cap_from_clearance(
    tree: Tree,
    node_index: int,
    q_node,
    is_free,
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
):
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return
    if tree.node_step_caps[node_index] is not None:
        return

    clearance_source = getattr(is_free, "strict_is_free", is_free)
    estimate_clearance = getattr(clearance_source, "estimate_clearance", None)
    if not callable(estimate_clearance):
        return

    clearance = float(estimate_clearance(q_node))
    minimum_clearance = float(getattr(clearance_source, "minimum_clearance", 0.0))
    clearance_margin = max(clearance - minimum_clearance, 0.0)
    min_step_size, max_step_size = _resolve_step_bounds(step_size, adaptive_step_config)
    local_step_cap = min_step_size + float(adaptive_step_config.clearance_step_cap_gain) * clearance_margin
    tree.node_step_caps[node_index] = float(np.clip(local_step_cap, min_step_size, max_step_size))

def _update_node_scale_after_success(
    tree: Tree,
    node_index: int,
    candidate_step_size: float,
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
) -> float:
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return 1.0

    nominal_step_size = max(1e-12, float(step_size))
    if nominal_step_size <= 1e-12:
        used_scale = 1.0
    else:
        used_scale = _clip_node_step_scale(
            float(candidate_step_size) / nominal_step_size,
            adaptive_step_config,
        )

    current_scale = float(tree.node_step_scales[node_index])
    alpha = float(adaptive_step_config.successful_step_ema_alpha)
    updated_scale = (1.0 - alpha) * current_scale + alpha * used_scale
    if used_scale >= current_scale - 1e-6:
        updated_scale *= float(adaptive_step_config.step_scale_growth_factor)
    updated_scale = _clip_node_step_scale(updated_scale, adaptive_step_config)
    tree.node_step_scales[node_index] = updated_scale
    tree.node_failure_counts[node_index] = max(0, int(tree.node_failure_counts[node_index]) - 1)
    tree.node_sample_domain_scales[node_index] = _clip_sample_domain_scale(
        float(tree.node_sample_domain_scales[node_index])
        * float(adaptive_step_config.sample_domain_growth_factor),
        adaptive_step_config,
    )
    return updated_scale

def _apply_node_failure_feedback(
    tree: Tree,
    node_index: int,
    candidate_step_size: float,
    q_node,
    is_free,
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
):
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return

    _maybe_update_node_step_cap_from_clearance(
        tree,
        node_index,
        q_node,
        is_free,
        step_size,
        adaptive_step_config,
    )
    nominal_step_size = max(1e-12, float(step_size))
    target_scale = _clip_node_step_scale(
        float(candidate_step_size) / nominal_step_size,
        adaptive_step_config,
    )
    shrunken_scale = min(
        float(tree.node_step_scales[node_index]) * float(adaptive_step_config.step_scale_shrink_factor),
        target_scale * float(adaptive_step_config.step_scale_shrink_factor),
    )
    tree.node_step_scales[node_index] = _clip_node_step_scale(
        shrunken_scale,
        adaptive_step_config,
    )
    tree.node_failure_counts[node_index] = int(tree.node_failure_counts[node_index]) + 1
    tree.node_sample_domain_scales[node_index] = _clip_sample_domain_scale(
        float(tree.node_sample_domain_scales[node_index])
        * float(adaptive_step_config.sample_domain_shrink_factor),
        adaptive_step_config,
    )

def _compute_connect_step_budget(
    tree: Tree,
    node_index: int,
    step_size: float,
    max_connect_steps: int,
    adaptive_step_config: AdaptiveStepConfig | None,
) -> int:
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return int(max_connect_steps)

    min_step_size, max_step_size = _resolve_step_bounds(step_size, adaptive_step_config)
    local_step_size = _compute_node_step_size(
        tree,
        node_index,
        step_size,
        adaptive_step_config,
    )
    if max_step_size <= min_step_size + 1e-12:
        openness = 1.0
    else:
        openness = (local_step_size - min_step_size) / (max_step_size - min_step_size)
    openness = float(np.clip(openness, 0.0, 1.0))
    failure_penalty = 1.0 / (1.0 + 0.5 * float(tree.node_failure_counts[node_index]))
    domain_factor = min(1.0, float(tree.node_sample_domain_scales[node_index]))
    budget_scale = (0.25 + 0.75 * openness) * failure_penalty * domain_factor
    connect_budget = int(round(float(max_connect_steps) * budget_scale))
    minimum_budget = min(int(max_connect_steps), int(adaptive_step_config.min_connect_steps))
    return max(
        minimum_budget,
        min(int(max_connect_steps), connect_budget),
    )

def _find_nearest_tree_node(
    tree: Tree,
    q,
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
) -> int:
    distances = [float(np.linalg.norm(n - q)) for n in tree.nodes]
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return int(np.argmin(distances))

    eligible_indices = []
    for idx, dist in enumerate(distances):
        local_radius = _compute_node_sample_domain_radius(
            tree,
            idx,
            step_size,
            adaptive_step_config,
        )
        if dist <= local_radius:
            eligible_indices.append(idx)
    if not eligible_indices:
        return int(np.argmin(distances))
    return min(eligible_indices, key=lambda idx: distances[idx])

def _sample_near_failure_nodes(
    rng,
    trees: tuple[Tree, Tree],
    step_size: float,
    adaptive_step_config: AdaptiveStepConfig | None,
    joints_lower_limits,
    joints_upper_limits,
):
    if adaptive_step_config is None or not adaptive_step_config.enabled:
        return None

    candidate_nodes = []
    candidate_weights = []
    for tree in trees:
        for node_index, q_node in enumerate(tree.nodes):
            failure_count = int(tree.node_failure_counts[node_index])
            local_step_cap = tree.node_step_caps[node_index]
            if failure_count <= 0 and local_step_cap is None:
                continue
            local_step_size = _compute_node_step_size(
                tree,
                node_index,
                step_size,
                adaptive_step_config,
            )
            weight = float(failure_count)
            if local_step_cap is not None:
                weight += 1.0
            candidate_nodes.append((np.asarray(q_node, dtype=float), local_step_size))
            candidate_weights.append(weight)

    if not candidate_nodes:
        return None

    weights = np.asarray(candidate_weights, dtype=float)
    weights /= np.sum(weights)
    sample_index = int(rng.choice(len(candidate_nodes), p=weights))
    q_center, local_step_size = candidate_nodes[sample_index]
    sigma = max(
        1e-4,
        float(adaptive_step_config.failure_node_sample_sigma_scale) * float(local_step_size),
    )
    q = q_center + rng.normal(0.0, sigma, size=q_center.shape)
    return np.clip(q, joints_lower_limits, joints_upper_limits)

def _path_is_dense_free(
    path,
    is_free,
    edge_resolution: float,
    deadline_s: float | None = None,
) -> bool:
    for i in range(len(path) - 1):
        if not edge_is_free(
            is_free,
            path[i],
            path[i + 1],
            resolution=edge_resolution,
            deadline_s=deadline_s,
        ):
            return False
    return True

def extend_tree(
    tree: Tree,
    q_target,
    is_free,
    step_size,
    edge_resolution,
    joints_lower_limits,
    joints_upper_limits,
    adaptive_step_config: AdaptiveStepConfig | None = None,
    deadline_s: float | None = None,
):
    """
    Try to extend tree by one step toward q_target.
    Returns (status, new_index_or_None).
    """

    q_target = np.asarray(q_target, dtype=float)
    if deadline_s is not None and time.perf_counter() > float(deadline_s):
        raise TimeoutError("RRT-Connect timed out before finding a path.")
    index_near = _find_nearest_tree_node(
        tree,
        q_target,
        step_size,
        adaptive_step_config,
    )
    q_near = tree.nodes[index_near]

    candidate_step_size = _compute_node_step_size(
        tree,
        index_near,
        step_size,
        adaptive_step_config,
    )
    min_step_size, _ = _resolve_step_bounds(step_size, adaptive_step_config)
    attempt_count = 1
    if adaptive_step_config is not None and adaptive_step_config.enabled:
        attempt_count = max(1, int(adaptive_step_config.max_backoff_trials) + 1)

    q_new = None
    for attempt_idx in range(attempt_count):
        q_new = find_new_node(q_near, q_target, candidate_step_size)
        q_new = np.clip(q_new, joints_lower_limits, joints_upper_limits)

        if np.linalg.norm(q_new - q_near) <= 1e-12:
            break
        if edge_is_free(
            is_free,
            q_near,
            q_new,
            resolution=edge_resolution,
            deadline_s=deadline_s,
        ):
            inherited_scale = _update_node_scale_after_success(
                tree,
                index_near,
                candidate_step_size,
                step_size,
                adaptive_step_config,
            )
            tree.nodes.append(q_new)
            tree.parent.append(index_near)
            tree.node_step_scales.append(inherited_scale)
            tree.node_sample_domain_scales.append(
                float(tree.node_sample_domain_scales[index_near])
            )
            tree.node_failure_counts.append(0)
            tree.node_step_caps.append(None)
            new_idx = len(tree.nodes) - 1

            # REACHED if we landed exactly on the target
            if np.linalg.norm(q_new - q_target) < 1e-8:
                return REACHED, new_idx
            return ADVANCED, new_idx

        if adaptive_step_config is None or not adaptive_step_config.enabled:
            break
        _apply_node_failure_feedback(
            tree,
            index_near,
            candidate_step_size,
            q_near,
            is_free,
            step_size,
            adaptive_step_config,
        )
        candidate_step_size = min(
            _compute_node_step_size(
                tree,
                index_near,
                step_size,
                adaptive_step_config,
            ),
            candidate_step_size * float(adaptive_step_config.backoff_factor),
        )
        if candidate_step_size < min_step_size:
            candidate_step_size = min_step_size

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
    deadline_s: float | None = None,
):
    """
    Greedily extend tree toward q_target until TRAPPED or REACHED.
    Returns (status, last_index or None if failed).
    """
    last_index = None
    connect_budget = int(max_connect_steps)
    if adaptive_step_config is not None and adaptive_step_config.enabled:
        near_index = _find_nearest_tree_node(
            tree,
            np.asarray(q_target, dtype=float),
            step_size,
            adaptive_step_config,
        )
        connect_budget = _compute_connect_step_budget(
            tree,
            near_index,
            step_size,
            max_connect_steps,
            adaptive_step_config,
        )
    for i in range(connect_budget):
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            raise TimeoutError("RRT-Connect timed out before finding a path.")
        status, index = extend_tree(
            tree, q_target, is_free, step_size, edge_resolution,
            joints_lower_limits, joints_upper_limits,
            adaptive_step_config=adaptive_step_config,
            deadline_s=deadline_s,
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
        search_is_free=None,
        step_size = 0.1,
        goal_sample_rate = 0.1,
        line_sample_rate=0.55,
        sigma_line=0.20,
        max_iters=30000,
        edge_resolution=0.01,
        max_sample_tries=30,
        max_connect_steps=10000,
        final_validation_edge_resolution=None,
        deadline_s=None,
        random_seed=None,
        adaptive_step_config: AdaptiveStepConfig | None = None,
        return_goal_index: bool = False,
):
    """
    Runs search-only RRT-connect from start to one goal or a set of goals.

    If `return_goal_index` is True and multiple goals are provided, returns
    `(path, selected_goal_index)`.
    """

    rng = np.random.default_rng(random_seed)
    search_start_s = time.perf_counter()
    telemetry = RRTConnectTelemetry()

    def _raise_with_telemetry(exc_type, message: str):
        telemetry.finalize(search_start_s)
        raise exc_type(f"{message} {telemetry.summary()}")

    q_start = np.asarray(q_start, dtype=float)
    joints_lower_limits = np.asarray(joints_lower_limits)
    joints_upper_limits = np.asarray(joints_upper_limits)
    if np.asarray(q_goal).ndim == 1:
        candidate_goals = [(0, np.asarray(q_goal, dtype=float))]
    else:
        candidate_goals = [
            (goal_index, np.asarray(goal_config, dtype=float))
            for goal_index, goal_config in enumerate(q_goal)
        ]

    _validate_adaptive_step_config(adaptive_step_config)
    final_validation_edge_resolution = _resolve_validation_resolution(
        edge_resolution,
        final_validation_edge_resolution,
    )
    if search_is_free is None:
        search_is_free = getattr(is_free, "growth_is_free", is_free)

    T_start = Tree(q_start, adaptive_step_config=adaptive_step_config)
    feasible_goals = []

    # check for feasibility:
    if not is_free(q_start):
        _raise_with_telemetry(RuntimeError, "q_start is not a feasible configuration.")
    for goal_index, q_goal_candidate in candidate_goals:
        if is_free(q_goal_candidate):
            feasible_goals.append((goal_index, q_goal_candidate))
        elif len(candidate_goals) == 1:
            _raise_with_telemetry(RuntimeError, "q_goal is not a feasible configuration.")
    if not feasible_goals:
        _raise_with_telemetry(RuntimeError, "No feasible q_goal candidates were provided.")
    telemetry.feasible_goals = len(feasible_goals)

    T_goal = Tree(feasible_goals[0][1], adaptive_step_config=adaptive_step_config)
    goal_root_index_to_goal_index = {0: feasible_goals[0][0]}
    for goal_index, q_goal_candidate in feasible_goals[1:]:
        append_tree_root(T_goal, q_goal_candidate)
        goal_root_index_to_goal_index[len(T_goal.nodes) - 1] = goal_index

    T_start_origin = "start"
    T_goal_origin = "goal"

    for goal_index, q_goal_candidate in feasible_goals:
        telemetry.direct_edge_checks += 1
        try:
            direct_edge_growth_free = edge_is_free(
                search_is_free,
                q_start,
                q_goal_candidate,
                resolution=edge_resolution,
                deadline_s=deadline_s,
            )
        except TimeoutError as exc:
            _raise_with_telemetry(TimeoutError, str(exc))
        if direct_edge_growth_free:
            telemetry.direct_edge_growth_successes += 1
            direct_path = [q_start.copy(), q_goal_candidate.copy()]
            try:
                direct_path_valid = _path_is_dense_free(
                    direct_path,
                    is_free,
                    edge_resolution=final_validation_edge_resolution,
                    deadline_s=deadline_s,
                )
            except TimeoutError as exc:
                _raise_with_telemetry(TimeoutError, str(exc))
            if direct_path_valid:
                telemetry.direct_edge_validation_successes += 1
                if return_goal_index:
                    return direct_path, goal_index
                return direct_path
            telemetry.direct_edge_validation_failures += 1

    # define function for sampling since we call it twice every time for each tree (start & goal)
    def sample(active_tree: Tree, passive_tree: Tree):
        """Mixture sampler from original RRT sampler"""
        r = rng.random()
        failure_node_sample_rate = 0.0
        if adaptive_step_config is not None and adaptive_step_config.enabled:
            failure_node_sample_rate = float(adaptive_step_config.failure_node_sample_rate)
        sampled_goal_index = int(rng.integers(len(feasible_goals)))
        sampled_goal = feasible_goals[sampled_goal_index][1]

        if r < goal_sample_rate:
            q = sampled_goal.copy()
        elif r < goal_sample_rate + line_sample_rate:
            # sample bias near the straight-line between start and goal (with added noise)
            u = rng.random()
            q = (
                (1 - u) * q_start
                + u * sampled_goal
                + rng.normal(0.0, sigma_line, size=q_start.shape)
            )
            q = np.clip(q, joints_lower_limits, joints_upper_limits)
        elif r < goal_sample_rate + line_sample_rate + failure_node_sample_rate:
            q = _sample_near_failure_nodes(
                rng,
                (active_tree, passive_tree),
                step_size,
                adaptive_step_config,
                joints_lower_limits,
                joints_upper_limits,
            )
            if q is None:
                q = rng.uniform(joints_lower_limits, joints_upper_limits)
        else:
            q = rng.uniform(joints_lower_limits, joints_upper_limits)

        return q

    for iter in range(max_iters):
        if deadline_s is not None and time.perf_counter() > float(deadline_s):
            _raise_with_telemetry(TimeoutError, "RRT-Connect timed out before finding a path.")
        telemetry.iterations = int(iter) + 1

        # Sample and reject samples that are in collision
        q_rand = None
        for i in range(max_sample_tries):
            if deadline_s is not None and time.perf_counter() > float(deadline_s):
                _raise_with_telemetry(TimeoutError, "RRT-Connect timed out before finding a path.")
            telemetry.sample_attempts += 1
            cand = sample(T_start, T_goal)
            if search_is_free(cand):
                q_rand = cand
                telemetry.sample_successes += 1
                break
            telemetry.sample_rejections += 1
        if q_rand is None:
            continue

        # Extend active tree one step toward sample
        try:
            status_start_branch, idx_start_branch = extend_tree(
                T_start, q_rand, search_is_free, step_size, edge_resolution,
                joints_lower_limits, joints_upper_limits,
                adaptive_step_config=adaptive_step_config,
                deadline_s=deadline_s,
            )
        except TimeoutError as exc:
            _raise_with_telemetry(TimeoutError, str(exc))
        if status_start_branch != TRAPPED:
            telemetry.extend_advanced += 1
            q_new = T_start.nodes[idx_start_branch]

            # Greedily connect the other tree toward the new node
            try:
                status_goal_branch, idx_goal_branch = connect(
                    T_goal, q_new, search_is_free, step_size, edge_resolution,
                    joints_lower_limits, joints_upper_limits,
                    max_connect_steps=max_connect_steps,
                    adaptive_step_config=adaptive_step_config,
                    deadline_s=deadline_s,
                )
            except TimeoutError as exc:
                _raise_with_telemetry(TimeoutError, str(exc))

            if status_goal_branch == REACHED:
                telemetry.connect_reached += 1
                if T_start_origin == "start":
                    path_start_to_conn = trace_path(T_start, idx_start_branch)
                    path_goal_to_conn, goal_root_index = trace_path_with_root_index(
                        T_goal,
                        idx_goal_branch,
                    )
                else:
                    path_goal_to_conn, goal_root_index = trace_path_with_root_index(
                        T_start,
                        idx_start_branch,
                    )
                    path_start_to_conn = trace_path(T_goal, idx_goal_branch)
                selected_goal_index = goal_root_index_to_goal_index[goal_root_index]

                # path_goal_to_conn is root(goal) -> ... -> conn, so reverse it to conn -> ... -> goal
                path = path_start_to_conn + path_goal_to_conn[::-1][1:]
                telemetry.candidate_paths_found += 1
                try:
                    path_valid = _path_is_dense_free(
                        path,
                        is_free,
                        edge_resolution=final_validation_edge_resolution,
                        deadline_s=deadline_s,
                    )
                except TimeoutError as exc:
                    _raise_with_telemetry(TimeoutError, str(exc))
                if not path_valid:
                    telemetry.candidate_paths_rejected_final_validation += 1
                    continue
                telemetry.finalize(search_start_s)
                print(
                    f"RRT-Connect succeeded in {telemetry.iterations} iterations. "
                    f"Raw path length: {len(path)}"
                )
                if return_goal_index:
                    return path, selected_goal_index
                return path
            elif status_goal_branch == TRAPPED:
                telemetry.connect_trapped += 1
        else:
            telemetry.extend_trapped += 1
        if status_start_branch != TRAPPED and status_goal_branch == ADVANCED:
            telemetry.connect_advanced += 1

        # Swap roles each iteration to advance one another each time
        T_start, T_goal = T_goal, T_start
        T_start_origin, T_goal_origin = T_goal_origin, T_start_origin

    _raise_with_telemetry(RuntimeError, "RRT-Connect failed.")
