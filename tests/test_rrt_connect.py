import time

import numpy as np

from src.planning.rrt_connect import (
    ADVANCED,
    TRAPPED,
    AdaptiveStepConfig,
    RRTConnectConfig,
    Tree,
    extend_tree,
    postprocess_rrt_path,
    rrt_connect_plan,
)


def _make_backoff_checker():
    def is_free(q):
        q = np.asarray(q, dtype=float)
        x = float(q[0])
        y = float(q[1])
        if 0.45 <= x <= 0.70 and abs(y) <= 0.15:
            return False
        return True

    is_free.estimate_clearance = lambda q: 0.05
    is_free.minimum_clearance = 0.0
    is_free.strict_is_free = is_free
    return is_free


def _make_trapped_checker():
    def is_free(q):
        q = np.asarray(q, dtype=float)
        return float(q[0]) <= 0.20

    is_free.estimate_clearance = lambda q: 0.01
    is_free.minimum_clearance = 0.0
    is_free.strict_is_free = is_free
    return is_free


def test_backoff_success_updates_node_scale_and_child_inherits():
    config = AdaptiveStepConfig(
        enabled=True,
        min_step_size=0.10,
        max_step_size=0.90,
        max_backoff_trials=3,
        backoff_factor=0.65,
        successful_step_ema_alpha=1.0,
        step_scale_growth_factor=1.0,
        step_scale_shrink_factor=0.70,
        initial_node_scale=1.0,
        node_scale_min=0.10,
        node_scale_max=1.0,
    )
    tree = Tree(np.array([0.0, 0.0]), adaptive_step_config=config)
    checker = _make_backoff_checker()

    status, new_index = extend_tree(
        tree=tree,
        q_target=np.array([1.0, 0.0]),
        is_free=checker,
        step_size=0.90,
        edge_resolution=0.05,
        joints_lower_limits=np.array([-2.0, -2.0]),
        joints_upper_limits=np.array([2.0, 2.0]),
        adaptive_step_config=config,
    )

    assert status == ADVANCED
    assert new_index is not None
    assert tree.node_step_scales[0] < 0.50
    assert tree.node_step_scales[new_index] == tree.node_step_scales[0]


def test_failure_shrinks_immediately_and_sets_clearance_cap():
    config = AdaptiveStepConfig(
        enabled=True,
        min_step_size=0.10,
        max_step_size=0.80,
        max_backoff_trials=0,
        backoff_factor=0.65,
        successful_step_ema_alpha=1.0,
        step_scale_growth_factor=1.0,
        step_scale_shrink_factor=0.50,
        initial_node_scale=1.0,
        node_scale_min=0.10,
        node_scale_max=1.0,
        clearance_step_cap_gain=10.0,
    )
    tree = Tree(np.array([0.0, 0.0]), adaptive_step_config=config)
    checker = _make_trapped_checker()

    status, new_index = extend_tree(
        tree=tree,
        q_target=np.array([1.0, 0.0]),
        is_free=checker,
        step_size=0.80,
        edge_resolution=0.05,
        joints_lower_limits=np.array([-2.0, -2.0]),
        joints_upper_limits=np.array([2.0, 2.0]),
        adaptive_step_config=config,
    )

    assert status == TRAPPED
    assert new_index is None
    assert tree.node_step_scales[0] < 1.0
    assert tree.node_step_caps[0] is not None


def test_rrt_direct_connect_returns_two_point_path():
    def is_free(q):
        return True

    path = rrt_connect_plan(
        q_start=np.array([0.0, 0.0]),
        q_goal=np.array([1.0, 0.0]),
        is_free=is_free,
        joints_lower_limits=np.array([-2.0, -2.0]),
        joints_upper_limits=np.array([2.0, 2.0]),
        step_size=0.50,
        goal_sample_rate=0.10,
        line_sample_rate=0.10,
        max_iters=20,
        edge_resolution=0.05,
        random_seed=0,
        adaptive_step_config=None,
    )

    assert len(path) == 2
    assert np.allclose(path[0], np.array([0.0, 0.0]))
    assert np.allclose(path[1], np.array([1.0, 0.0]))


def test_rrt_multi_goal_returns_selected_goal_index():
    def is_free(q):
        q = np.asarray(q, dtype=float)
        x = float(q[0])
        y = float(q[1])
        if 0.45 <= x <= 0.70 and abs(y) <= 0.15:
            return False
        return True

    path, goal_index = rrt_connect_plan(
        q_start=np.array([0.0, 0.0]),
        q_goal=[
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ],
        is_free=is_free,
        joints_lower_limits=np.array([-2.0, -2.0]),
        joints_upper_limits=np.array([2.0, 2.0]),
        step_size=0.50,
        goal_sample_rate=0.10,
        line_sample_rate=0.10,
        max_iters=20,
        edge_resolution=0.05,
        random_seed=0,
        adaptive_step_config=None,
        return_goal_index=True,
    )

    assert goal_index == 1
    assert np.allclose(path[0], np.array([0.0, 0.0]))
    assert np.allclose(path[-1], np.array([0.0, 1.0]))


def test_postprocess_rrt_path_shortcuts_with_budgeted_random_attempts():
    def strict_is_free(q):
        return True

    def growth_is_free(q):
        return True

    growth_is_free.strict_is_free = strict_is_free
    strict_is_free.growth_is_free = growth_is_free

    raw_path = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.6]),
        np.array([1.0, 0.0]),
    ]
    config = RRTConnectConfig(
        enable_shortcut=True,
        shortcut_edge_resolution=0.20,
        shortcut_max_attempts=8,
        shortcut_time_budget_s=1.0,
        final_validation_edge_resolution=0.05,
        random_seed=0,
        adaptive_step_config=None,
    )

    path = postprocess_rrt_path(raw_path, strict_is_free, planner_config=config)

    assert len(path) == 2
    assert np.allclose(path[0], raw_path[0])
    assert np.allclose(path[-1], raw_path[-1])


def test_postprocess_rrt_path_falls_back_to_raw_path_when_shortcut_fails_strict_validation():
    raw_path = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    ]

    def strict_is_free(q):
        q = np.asarray(q, dtype=float)
        x = float(q[0])
        y = float(q[1])
        return not (0.25 < x < 0.75 and 0.25 < y < 0.75)

    def growth_is_free(q):
        return True

    growth_is_free.strict_is_free = strict_is_free
    strict_is_free.growth_is_free = growth_is_free

    config = RRTConnectConfig(
        enable_shortcut=True,
        shortcut_edge_resolution=0.20,
        shortcut_max_attempts=8,
        shortcut_time_budget_s=1.0,
        final_validation_edge_resolution=0.05,
        random_seed=0,
        adaptive_step_config=None,
    )

    path = postprocess_rrt_path(raw_path, strict_is_free, planner_config=config)

    assert len(path) == len(raw_path)
    for q_result, q_expected in zip(path, raw_path):
        assert np.allclose(q_result, q_expected)


def test_rrt_failure_message_includes_telemetry():
    def is_free(q):
        q = np.asarray(q, dtype=float)
        x = float(q[0])
        y = float(q[1])
        return not (0.40 <= x <= 0.60 and abs(y) <= 0.20)

    try:
        rrt_connect_plan(
            q_start=np.array([0.0, 0.0]),
            q_goal=np.array([1.0, 0.0]),
            is_free=is_free,
            joints_lower_limits=np.array([-2.0, -2.0]),
            joints_upper_limits=np.array([2.0, 2.0]),
            step_size=0.50,
            goal_sample_rate=0.10,
            line_sample_rate=0.10,
            max_iters=0,
            edge_resolution=0.05,
            random_seed=0,
            adaptive_step_config=None,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected RRT-Connect to fail.")

    assert "RRT telemetry:" in message
    assert "direct_edge_checks=1" in message
    assert "iterations=0" in message


def test_rrt_timeout_message_includes_telemetry():
    def is_free(q):
        q = np.asarray(q, dtype=float)
        x = float(q[0])
        y = float(q[1])
        return not (0.40 <= x <= 0.60 and abs(y) <= 0.20)

    try:
        rrt_connect_plan(
            q_start=np.array([0.0, 0.0]),
            q_goal=np.array([1.0, 0.0]),
            is_free=is_free,
            joints_lower_limits=np.array([-2.0, -2.0]),
            joints_upper_limits=np.array([2.0, 2.0]),
            step_size=0.50,
            goal_sample_rate=0.10,
            line_sample_rate=0.10,
            max_iters=20,
            edge_resolution=0.05,
            deadline_s=time.perf_counter() - 1e-3,
            random_seed=0,
            adaptive_step_config=None,
        )
    except TimeoutError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected RRT-Connect to time out.")

    assert "RRT telemetry:" in message
    assert "elapsed_s=" in message


def test_rrt_search_checker_override_disables_relaxed_growth_shortcut():
    def strict_is_free(q):
        q = np.asarray(q, dtype=float)
        x = float(q[0])
        return not (0.25 < x < 0.75)

    def growth_is_free(q):
        return True

    strict_is_free.growth_is_free = growth_is_free
    growth_is_free.strict_is_free = strict_is_free

    try:
        rrt_connect_plan(
            q_start=np.array([0.0]),
            q_goal=np.array([1.0]),
            is_free=strict_is_free,
            search_is_free=strict_is_free,
            joints_lower_limits=np.array([-2.0]),
            joints_upper_limits=np.array([2.0]),
            max_iters=0,
            edge_resolution=0.05,
            adaptive_step_config=None,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected RRT-Connect to fail.")

    assert "direct_edge_growth_successes=0" in message
    assert "direct_edge_validation_failures=0" in message


if __name__ == "__main__":
    test_backoff_success_updates_node_scale_and_child_inherits()
    test_failure_shrinks_immediately_and_sets_clearance_cap()
    test_rrt_direct_connect_returns_two_point_path()
    test_rrt_multi_goal_returns_selected_goal_index()
    test_postprocess_rrt_path_shortcuts_with_budgeted_random_attempts()
    test_postprocess_rrt_path_falls_back_to_raw_path_when_shortcut_fails_strict_validation()
    test_rrt_failure_message_includes_telemetry()
    test_rrt_timeout_message_includes_telemetry()
    test_rrt_search_checker_override_disables_relaxed_growth_shortcut()
