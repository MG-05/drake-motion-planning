import numpy as np

from src.planning.rrt_connect import (
    ADVANCED,
    TRAPPED,
    AdaptiveStepConfig,
    Tree,
    extend_tree,
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
        enable_shortcut=False,
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
        enable_shortcut=False,
        random_seed=0,
        adaptive_step_config=None,
        return_goal_index=True,
    )

    assert goal_index == 1
    assert np.allclose(path[0], np.array([0.0, 0.0]))
    assert np.allclose(path[-1], np.array([0.0, 1.0]))


if __name__ == "__main__":
    test_backoff_success_updates_node_scale_and_child_inherits()
    test_failure_shrinks_immediately_and_sets_clearance_cap()
    test_rrt_direct_connect_returns_two_point_path()
    test_rrt_multi_goal_returns_selected_goal_index()
