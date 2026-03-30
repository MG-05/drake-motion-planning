import numpy as np

from src.manipulation.manipulation_fsm import build_axial_clearance_path


def test_axial_clearance_path_stops_on_clearance_threshold():
    result = build_axial_clearance_path(
        q_anchor=np.array([0.0]),
        step_m=0.02,
        max_pullback_m=0.10,
        solve_step=lambda pullback_m, q_seed, step_index: np.array([pullback_m]),
        state_is_free=lambda q: True,
        edge_is_free_fn=lambda q0, q1: True,
        estimate_clearance_fn=lambda q: float(q[0]),
        target_clearance_m=0.05,
    )

    assert result.stop_reason == "clearance"
    assert len(result.path) == 4
    assert np.allclose(result.path[-1], np.array([0.06]))
    assert abs(float(result.final_clearance_m) - 0.06) <= 1e-12


def test_axial_clearance_path_stops_on_predicate_before_clearance():
    result = build_axial_clearance_path(
        q_anchor=np.array([0.0]),
        step_m=0.02,
        max_pullback_m=0.10,
        solve_step=lambda pullback_m, q_seed, step_index: np.array([pullback_m]),
        state_is_free=lambda q: True,
        edge_is_free_fn=lambda q0, q1: True,
        estimate_clearance_fn=lambda q: float(q[0]),
        target_clearance_m=0.10,
        stop_predicate=lambda q: float(q[0]) >= 0.04,
    )

    assert result.stop_reason == "predicate"
    assert len(result.path) == 3
    assert np.allclose(result.path[-1], np.array([0.04]))


def test_axial_clearance_path_skips_failed_steps_and_continues():
    def solve_step(pullback_m, q_seed, step_index):
        if abs(float(pullback_m) - 0.02) <= 1e-12:
            raise RuntimeError("synthetic IK failure")
        return np.array([pullback_m])

    result = build_axial_clearance_path(
        q_anchor=np.array([0.0]),
        step_m=0.02,
        max_pullback_m=0.10,
        solve_step=solve_step,
        state_is_free=lambda q: True,
        edge_is_free_fn=lambda q0, q1: True,
        estimate_clearance_fn=lambda q: float(q[0]),
        target_clearance_m=0.03,
    )

    assert result.stop_reason == "clearance"
    assert len(result.path) == 2
    assert np.allclose(result.path[-1], np.array([0.04]))


def test_axial_clearance_path_can_stop_at_anchor_without_solving():
    solve_calls = {"count": 0}

    def solve_step(pullback_m, q_seed, step_index):
        solve_calls["count"] += 1
        return np.array([pullback_m])

    result = build_axial_clearance_path(
        q_anchor=np.array([0.08]),
        step_m=0.02,
        max_pullback_m=0.10,
        solve_step=solve_step,
        state_is_free=lambda q: True,
        edge_is_free_fn=lambda q0, q1: True,
        estimate_clearance_fn=lambda q: float(q[0]),
        target_clearance_m=0.05,
    )

    assert result.stop_reason == "clearance"
    assert len(result.path) == 1
    assert np.allclose(result.path[-1], np.array([0.08]))
    assert solve_calls["count"] == 0


if __name__ == "__main__":
    test_axial_clearance_path_stops_on_clearance_threshold()
    test_axial_clearance_path_stops_on_predicate_before_clearance()
    test_axial_clearance_path_skips_failed_steps_and_continues()
    test_axial_clearance_path_can_stop_at_anchor_without_solving()
