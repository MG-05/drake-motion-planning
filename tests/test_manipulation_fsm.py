import numpy as np
from pydrake.math import RigidTransform

from src.manipulation.manipulation_fsm import (
    AxialClearancePathResult,
    ManipulationFSM,
    ManipulationOptions,
    build_axial_clearance_path,
)


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


def test_carry_escape_prefix_prioritizes_home_staging_over_clearance_stop():
    fsm = ManipulationFSM.__new__(ManipulationFSM)
    fsm.options = ManipulationOptions(
        carry_escape_try_home_staging=True,
        carry_escape_clearance_threshold_m=0.031,
        carry_escape_open_clearance_margin_m=0.02,
    )

    q_home = np.full(7, 10.0)
    q_postgrasp_retreat = np.zeros(7)
    q_stage = np.full(7, 1.0)

    fsm.is_free_carry = lambda q: True

    def fake_edge_is_free(q_start, q_goal, is_free_fn, planning_deadline_s=None):
        return np.allclose(np.asarray(q_start, dtype=float), q_stage) and np.allclose(
            np.asarray(q_goal, dtype=float), q_home
        )

    fsm._strict_edge_is_free = fake_edge_is_free

    def fake_plan_axial_clearance_path(
        q_anchor,
        is_free_fn,
        *,
        clearance_source_fn=None,
        absolute_clearance_threshold_m,
        open_clearance_margin_m,
        max_pullback_m,
        step_m,
        planning_deadline_s=None,
        stop_predicate=None,
        soft_start_seed_base=0,
    ):
        assert clearance_source_fn is is_free_fn
        assert absolute_clearance_threshold_m is None
        assert open_clearance_margin_m is None
        assert stop_predicate is not None
        assert stop_predicate(q_anchor) is False
        assert stop_predicate(q_stage) is True
        return AxialClearancePathResult(
            path=[np.asarray(q_anchor, dtype=float).copy(), q_stage.copy()],
            final_clearance_m=0.05,
            stop_reason="predicate",
        )

    fsm._plan_axial_clearance_path = fake_plan_axial_clearance_path

    carry_prefix, q_carry_start = fsm._plan_carry_escape_prefix(
        q_home=q_home,
        q_postgrasp_retreat=q_postgrasp_retreat,
    )

    assert len(carry_prefix) == 3
    assert np.allclose(carry_prefix[0], q_postgrasp_retreat)
    assert np.allclose(carry_prefix[1], q_stage)
    assert np.allclose(carry_prefix[-1], q_home)
    assert np.allclose(q_carry_start, q_home)


def test_carry_escape_prefix_keeps_clearance_stop_when_home_staging_unavailable():
    fsm = ManipulationFSM.__new__(ManipulationFSM)
    fsm.options = ManipulationOptions(
        carry_escape_try_home_staging=True,
        carry_escape_clearance_threshold_m=0.031,
        carry_escape_open_clearance_margin_m=0.02,
    )

    q_home = np.full(7, 10.0)
    q_postgrasp_retreat = np.zeros(7)

    def fake_is_free_carry(q):
        return not np.allclose(np.asarray(q, dtype=float), q_home)

    fsm.is_free_carry = fake_is_free_carry
    fsm._strict_edge_is_free = lambda *args, **kwargs: False

    def fake_plan_axial_clearance_path(
        q_anchor,
        is_free_fn,
        *,
        clearance_source_fn=None,
        absolute_clearance_threshold_m,
        open_clearance_margin_m,
        max_pullback_m,
        step_m,
        planning_deadline_s=None,
        stop_predicate=None,
        soft_start_seed_base=0,
    ):
        assert clearance_source_fn is is_free_fn
        assert abs(float(absolute_clearance_threshold_m) - 0.031) <= 1e-12
        assert abs(float(open_clearance_margin_m) - 0.02) <= 1e-12
        return AxialClearancePathResult(
            path=[np.asarray(q_anchor, dtype=float).copy()],
            final_clearance_m=0.031,
            stop_reason="clearance",
        )

    fsm._plan_axial_clearance_path = fake_plan_axial_clearance_path

    carry_prefix, q_carry_start = fsm._plan_carry_escape_prefix(
        q_home=q_home,
        q_postgrasp_retreat=q_postgrasp_retreat,
    )

    assert len(carry_prefix) == 1
    assert np.allclose(carry_prefix[0], q_postgrasp_retreat)
    assert np.allclose(q_carry_start, q_postgrasp_retreat)


def test_drop_candidates_prioritize_nominal_and_translation_offsets_before_yaw():
    fsm = ManipulationFSM.__new__(ManipulationFSM)
    fsm.options = ManipulationOptions(
        drop_xy_offsets_m=((0.0, 0.0), (0.03, 0.0)),
        drop_z_offsets_m=(0.0, 0.03),
        drop_yaw_offsets_rad=(0.0, 0.1),
    )

    candidates = []
    for candidate_index, X_WG_candidate in fsm._iter_drop_candidates(RigidTransform()):
        candidates.append((candidate_index, X_WG_candidate))
        if len(candidates) == 6:
            break

    candidate_translations = [
        tuple(np.asarray(X_WG_candidate.translation(), dtype=float))
        for _, X_WG_candidate in candidates
    ]
    assert candidate_translations[:4] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.03),
        (0.03, 0.0, 0.0),
        (0.03, 0.0, 0.03),
    ]
    for _, X_WG_candidate in candidates[:4]:
        assert np.allclose(X_WG_candidate.rotation().matrix(), np.eye(3))
    assert not np.allclose(candidates[4][1].rotation().matrix(), np.eye(3))


def test_drop_rrt_seed_offsets_prefer_small_and_large_batch_defaults():
    fsm = ManipulationFSM.__new__(ManipulationFSM)
    fsm.options = ManipulationOptions(drop_rrt_seed_offsets=(2, 3, 4))

    assert fsm._ordered_drop_rrt_seed_offsets(batch_size=1) == (3, 2, 4)
    assert fsm._ordered_drop_rrt_seed_offsets(batch_size=5) == (3, 2, 4)
    assert fsm._ordered_drop_rrt_seed_offsets(batch_size=6) == (2, 3, 4)


def test_require_pose_reachable_returns_anchor_ik_solution():
    fsm = ManipulationFSM.__new__(ManipulationFSM)
    expected_q = np.arange(7, dtype=float)
    call_args = {}

    def fake_solve_anchor_pose_ik(
        X_WG_target,
        q_seed,
        planning_deadline_s=None,
        *,
        soft_start_seed_offset=0,
    ):
        call_args["translation"] = tuple(
            np.asarray(X_WG_target.translation(), dtype=float).reshape(3)
        )
        call_args["q_seed"] = np.asarray(q_seed, dtype=float).copy()
        call_args["planning_deadline_s"] = planning_deadline_s
        call_args["soft_start_seed_offset"] = soft_start_seed_offset
        return expected_q.copy()

    fsm._solve_anchor_pose_ik = fake_solve_anchor_pose_ik

    q_result = fsm._require_pose_reachable(
        pose_label="Drop",
        X_WG_target=RigidTransform([0.1, 0.2, 0.3]),
        q_seed=np.zeros(7),
        planning_deadline_s=12.5,
        soft_start_seed_offset=42,
    )

    assert np.allclose(q_result, expected_q)
    assert call_args["translation"] == (0.1, 0.2, 0.3)
    assert np.allclose(call_args["q_seed"], np.zeros(7))
    assert abs(float(call_args["planning_deadline_s"]) - 12.5) <= 1e-12
    assert call_args["soft_start_seed_offset"] == 42


def test_require_pose_reachable_raises_clear_not_in_reach_error():
    fsm = ManipulationFSM.__new__(ManipulationFSM)

    def fake_solve_anchor_pose_ik(*args, **kwargs):
        raise RuntimeError("synthetic IK failure")

    fsm._solve_anchor_pose_ik = fake_solve_anchor_pose_ik

    try:
        fsm._require_pose_reachable(
            pose_label="Drop",
            X_WG_target=RigidTransform([0.125, -0.05, 0.58]),
            q_seed=np.zeros(7),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected a reachability RuntimeError")

    assert message == "Drop location is not in reach (target position [0.125, -0.050, 0.580])."


def test_drop_preplace_uses_decoupled_clearance_source():
    fsm = ManipulationFSM.__new__(ManipulationFSM)
    fsm.options = ManipulationOptions(
        enable_drop_preplace=True,
        drop_preplace_clearance_threshold_m=0.05,
        drop_preplace_max_pullback_m=0.20,
        axial_pullback_step_m=0.015,
    )

    state_checker = lambda q: True
    clearance_checker = lambda q: True
    fsm.is_free_carry = state_checker
    fsm.drop_preplace_clearance_source = clearance_checker

    q_drop_candidate = np.arange(7, dtype=float)
    q_preplace = q_drop_candidate + 1.0
    captured = {}

    def fake_plan_axial_clearance_path(
        q_anchor,
        is_free_fn,
        *,
        clearance_source_fn=None,
        absolute_clearance_threshold_m,
        open_clearance_margin_m,
        max_pullback_m,
        step_m,
        planning_deadline_s=None,
        stop_predicate=None,
        soft_start_seed_base=0,
    ):
        captured["q_anchor"] = np.asarray(q_anchor, dtype=float).copy()
        captured["is_free_fn"] = is_free_fn
        captured["clearance_source_fn"] = clearance_source_fn
        captured["absolute_clearance_threshold_m"] = absolute_clearance_threshold_m
        captured["open_clearance_margin_m"] = open_clearance_margin_m
        captured["max_pullback_m"] = max_pullback_m
        captured["step_m"] = step_m
        captured["soft_start_seed_base"] = soft_start_seed_base
        return AxialClearancePathResult(
            path=[np.asarray(q_anchor, dtype=float).copy(), q_preplace.copy()],
            final_clearance_m=0.05,
            stop_reason="clearance",
        )

    fsm._plan_axial_clearance_path = fake_plan_axial_clearance_path

    q_transit_goal, insertion_path = fsm._find_drop_transit_goal(
        candidate_index=3,
        X_WG_drop_candidate=RigidTransform(),
        q_drop_candidate=q_drop_candidate,
    )

    assert np.allclose(captured["q_anchor"], q_drop_candidate)
    assert captured["is_free_fn"] is state_checker
    assert captured["clearance_source_fn"] is clearance_checker
    assert abs(float(captured["absolute_clearance_threshold_m"]) - 0.05) <= 1e-12
    assert abs(float(captured["open_clearance_margin_m"]) - 0.02) <= 1e-12
    assert abs(float(captured["max_pullback_m"]) - 0.20) <= 1e-12
    assert abs(float(captured["step_m"]) - 0.015) <= 1e-12
    assert captured["soft_start_seed_base"] == 10_300
    assert np.allclose(q_transit_goal, q_preplace)
    assert len(insertion_path) == 2
    assert np.allclose(insertion_path[0], q_preplace)
    assert np.allclose(insertion_path[1], q_drop_candidate)


if __name__ == "__main__":
    test_axial_clearance_path_stops_on_clearance_threshold()
    test_axial_clearance_path_stops_on_predicate_before_clearance()
    test_axial_clearance_path_skips_failed_steps_and_continues()
    test_axial_clearance_path_can_stop_at_anchor_without_solving()
    test_carry_escape_prefix_prioritizes_home_staging_over_clearance_stop()
    test_carry_escape_prefix_keeps_clearance_stop_when_home_staging_unavailable()
    test_drop_candidates_prioritize_nominal_and_translation_offsets_before_yaw()
    test_drop_rrt_seed_offsets_prefer_small_and_large_batch_defaults()
    test_require_pose_reachable_returns_anchor_ik_solution()
    test_require_pose_reachable_raises_clear_not_in_reach_error()
    test_drop_preplace_uses_decoupled_clearance_source()
