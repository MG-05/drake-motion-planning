# Adapted from: https://github.com/RobotLocomotion/drake/blob/master/examples/hardware_sim/hardware_sim.py
import argparse
import dataclasses as dc
import math
import sys
import typing
import webbrowser
from pathlib import Path

import numpy as np

from pydrake.common.yaml import yaml_load_typed
from pydrake.geometry import Meshcat, SceneGraphConfig
from pydrake.lcm import DrakeLcmParams
from pydrake.manipulation import (
    ApplyDriverConfigs,
    IiwaDriver,
    SchunkWsgDriver,
    ZeroForceDriver,
)
from pydrake.multibody.parsing import ModelDirective, ModelDirectives, ProcessModelDirectives
from pydrake.multibody.plant import AddMultibodyPlant, MultibodyPlantConfig
from pydrake.systems.analysis import ApplySimulatorConfig, Simulator, SimulatorConfig
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.lcm import ApplyLcmBusConfig
from pydrake.trajectories import PiecewisePolynomial
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig
from pydrake.math import RigidTransform, RollPitchYaw

# LCM imports
from drake import lcmt_iiwa_command, lcmt_schunk_wsg_command
from pydrake.lcm import DrakeLcm

from src.planning.collision import is_collision_free
from src.manipulation.grasp import GraspOptions
from src.manipulation.manipulation_fsm import ManipulationFSM, ManipulationOptions
from src.manipulation.pregrasp import get_floating_body, compute_pregrasp_pose_for_brick


@dc.dataclass
class Scenario:
    """
    Defines the YAML format for the scenario we want simulated.
    """
    random_seed: int = 0
    # max sim time
    simulation_duration: float = math.inf

    # Simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = SimulatorConfig(
        max_step_size=1e-3, accuracy=1e-2, target_realtime_rate=1.0
    )
    plant_config: MultibodyPlantConfig = MultibodyPlantConfig()
    scene_graph_config: SceneGraphConfig = SceneGraphConfig()

    # All elements of the simulation
    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # A mapping of {bus_name: lcm_paramaters} for LCM tranceivers to be used by sensors
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams())
    )

    # Specify where each model's actuation inputs come form
    model_drivers: typing.Mapping[
        str, typing.Union[IiwaDriver, SchunkWsgDriver, ZeroForceDriver]
    ] = dc.field(default_factory=dict)

    visualization: VisualizationConfig = VisualizationConfig()

def default_scenario_path() -> Path:
    """
    Returns the default scenario YAML path
    """
    repo_src = Path(__file__).resolve().parents[1]
    return repo_src / "configs" / "scenes" / "starter_env.yaml"


def _compute_iiwa_joint_limits(sim_plant, iiwa_instance):
    joint_names = [f"iiwa_joint_{i}" for i in range(1, 8)]
    lower = []
    upper = []
    for name in joint_names:
        joint = sim_plant.GetJointByName(name, iiwa_instance)
        lower.append(joint.position_lower_limits()[0])
        upper.append(joint.position_upper_limits()[0])
    return np.asarray(lower), np.asarray(upper)


def _build_trajectory_from_path(path, vmax=0.35):
    if len(path) == 0:
        raise ValueError("Path is empty")

    path = [np.asarray(q, dtype=float).reshape(7) for q in path]
    if len(path) == 1:
        times = np.array([0.0, 0.05])
        knots = np.vstack([path[0], path[0]]).T
        return PiecewisePolynomial.FirstOrderHold(times, knots), float(times[-1])

    times = [0.0]
    for i in range(len(path) - 1):
        dq = np.abs(path[i + 1] - path[i])
        seg_T = max(0.05, float(np.max(dq) / vmax))
        times.append(times[-1] + seg_T)

    times = np.asarray(times, dtype=float)
    knots = np.asarray(path, dtype=float).T
    return PiecewisePolynomial.FirstOrderHold(times, knots), float(times[-1])


def _publish_and_advance_segment(
    simulator,
    trajectory,
    duration,
    lcm,
    iiwa_cmd,
    wsg_cmd,
    gripper_target_mm,
    command_dt=0.01,
):
    t = 0.0
    while t < duration:
        t_next = min(t + command_dt, duration)
        iiwa_cmd.joint_position = trajectory.value(t_next).ravel()
        wsg_cmd.target_position_mm = float(gripper_target_mm)
        lcm.Publish(channel="IIWA_COMMAND", buffer=iiwa_cmd.encode())
        lcm.Publish(channel="SCHUNK_WSG_COMMAND", buffer=wsg_cmd.encode())
        simulator.AdvanceTo(simulator.get_context().get_time() + (t_next - t))
        t = t_next


def _hold_current_position(
    simulator,
    q_hold,
    hold_time,
    lcm,
    iiwa_cmd,
    wsg_cmd,
    gripper_target_mm,
    command_dt=0.01,
):
    q_hold = np.asarray(q_hold, dtype=float).reshape(7)
    t = 0.0
    while t < hold_time:
        t_next = min(t + command_dt, hold_time)
        iiwa_cmd.joint_position = q_hold
        wsg_cmd.target_position_mm = float(gripper_target_mm)
        lcm.Publish(channel="IIWA_COMMAND", buffer=iiwa_cmd.encode())
        lcm.Publish(channel="SCHUNK_WSG_COMMAND", buffer=wsg_cmd.encode())
        simulator.AdvanceTo(simulator.get_context().get_time() + (t_next - t))
        t = t_next


def _compute_wsg_planning_configs(q_wsg_open):
    """
    Build open / near-closed WSG planning configs in plant position units.
    """
    q_wsg_open = np.asarray(q_wsg_open, dtype=float).reshape(-1)
    # Keep sign convention from the model and shrink toward zero to approximate closed.
    q_wsg_closed = 0.02 * q_wsg_open
    return q_wsg_open.copy(), q_wsg_closed


def main():

    # sample parser arguments from source Drake code (slightly modified)
    parser = argparse.ArgumentParser(
        description="Run StarterEnv in Meshcat using Drake hardware_sim starter."
    )
    parser.add_argument("--scenario_file", type=Path, default=default_scenario_path())
    parser.add_argument("--scenario_name", type=str, default="StarterEnv")
    parser.add_argument("--duration", type=float, default=None, help="Override YAML simulation_duration.")
    parser.add_argument("--open", action="store_true", help="Open Meshcat in browser.")
    args = parser.parse_args()

    if not args.scenario_file.exists():
        print(f"ERROR - Scenario file not found: {args.scenario_file}", file=sys.stderr)
        return 2

    # Start Meshcat
    meshcat = Meshcat()
    print(f"[Meshcat] {meshcat.web_url()}")
    if args.open:
        webbrowser.open(meshcat.web_url())

    # Load scenario from YAML (top-level key = scenario_name)
    scenario = yaml_load_typed(
        schema=Scenario,
        filename=str(args.scenario_file),
        child_name=args.scenario_name,
        defaults=Scenario(),
    )

    # Build diagram
    builder = DiagramBuilder()

    # Create multibody plant and scene graph
    sim_plant, scene_graph = AddMultibodyPlant(
        plant_config=scenario.plant_config,
        scene_graph_config=scenario.scene_graph_config,
        builder=builder,
    )

    # Add models directives
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        plant=sim_plant,
    )

    sim_plant.Finalize()

    # Add LCM buses
    lcm_buses = ApplyLcmBusConfig(lcm_buses=scenario.lcm_buses, builder=builder)

    # Apply actuation inputs
    ApplyDriverConfigs(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        models_from_directives=added_models,
        lcm_buses=lcm_buses,
        builder=builder,
    )

    # Visualization to Meshcat
    ApplyVisualizationConfig(
        config=scenario.visualization,
        builder=builder,
        lcm_buses=lcm_buses,
        meshcat=meshcat,
        plant=sim_plant,
        scene_graph=scene_graph,
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.Initialize()

    root_context = simulator.get_mutable_context()
    plant_context = sim_plant.GetMyMutableContextFromRoot(root_context)

    # Place the foam brick only if it is a free body to avoid floating-base asserts.
    brick_instance = None
    brick_body = None
    if sim_plant.HasModelInstanceNamed("foam_brick"):
        brick_instance = sim_plant.GetModelInstanceByName("foam_brick")
        for body_index in sim_plant.GetBodyIndices(brick_instance):
            body = sim_plant.get_body(body_index)
            if body.is_floating():
                brick_body = body
                break

        if brick_body is None:
            print("Warning: foam_brick has no floating body; skipping free-body pose.")
        else:
            # Lower shelf -> [0.0, 0.15, 0.030],
            # Middle Shelf -> [0.0, 0.15, 0.30],
            # Top Shelf -> [0.0, 0.15, 0.56]
            X_WB = RigidTransform(
                RollPitchYaw(0.0, 0.0, math.radians(-90.0)).ToRotationMatrix(),
                [0.0, 0.15, 0.30],
            )
            sim_plant.SetFreeBodyPose(plant_context, brick_body, X_WB)
    else:
        raise RuntimeError("foam_brick model instance is required for grasp planning")

    iiwa = sim_plant.GetModelInstanceByName("iiwa")

    # Start Config
    q_start = sim_plant.GetPositions(plant_context, iiwa).copy()

    wsg = sim_plant.GetModelInstanceByName("wsg")
    q_wsg_open_plan = sim_plant.GetPositions(plant_context, wsg).copy()
    q_wsg_open_plan, q_wsg_closed_plan = _compute_wsg_planning_configs(q_wsg_open_plan)


    # Strict collision checker for transit motion (requires clearance margin).
    is_free = is_collision_free(
        plant=sim_plant,
        scene_graph=scene_graph,
        root_context=root_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        q_wsg_instance=q_wsg_closed_plan,
        min_clearance=0.01,
        pair_range=0.05,
    )
    # Relaxed checker for grasp approach (penetration-free, but allows near-contact).
    is_free_grasp = is_collision_free(
        plant=sim_plant,
        scene_graph=scene_graph,
        root_context=root_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        q_wsg_instance=q_wsg_open_plan,
        min_clearance=0.0,
        pair_range=0.05,
        ignore_model_instances=[brick_instance],
    )
    # Carry checker for post-grasp transport/drop planning.
    is_free_carry = is_collision_free(
        plant=sim_plant,
        scene_graph=scene_graph,
        root_context=root_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        q_wsg_instance=q_wsg_closed_plan,
        min_clearance=0.01,
        pair_range=0.08,
        extra_checked_model_instances=[brick_instance],
    )

    # -------------------------
    # Pre-grasp target for the foam brick
    # -------------------------
    brick_body = get_floating_body(sim_plant, brick_instance)

    X_WG_pregrasp = compute_pregrasp_pose_for_brick(
        plant=sim_plant,
        plant_context=plant_context,
        iiwa_instance=iiwa,
        brick_body=brick_body,
        fingertip_clearance_m=0.04,  # "few cm away"
        wsg_body_to_fingertips_m=0.14,  # practical constant; tune if needed
    )
    print("\n[Pregrasp] X_WG_pregrasp =", X_WG_pregrasp)

    # Drop target for place/release.
    drop_position_W = np.array([0.0, 0.15, 0.58])
    X_WG_drop = RigidTransform(X_WG_pregrasp.rotation(), drop_position_W)

    joints_lower_limits, joints_upper_limits = _compute_iiwa_joint_limits(sim_plant, iiwa)
    print(f"Is the start config collision free? {is_free(q_start)}")

    target_z = float(X_WG_pregrasp.translation()[2])
    ik_soft_starts = 16 if 0.20 <= target_z <= 0.39 else 10

    grasp_options = GraspOptions(
        ik_soft_starts=ik_soft_starts,
        ik_soft_start_sigma=0.08,
        ik_soft_start_seed=0,
    )
    fsm_options = ManipulationOptions(
        ik_soft_starts=ik_soft_starts,
        ik_soft_start_sigma=0.08,
        ik_soft_start_seed=0,
        grasp_options=grasp_options,
    )
    fsm = ManipulationFSM(
        plant=sim_plant,
        root_context_current=root_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        is_free=is_free,
        joints_lower_limits=joints_lower_limits,
        joints_upper_limits=joints_upper_limits,
        is_free_grasp=is_free_grasp,
        is_free_carry=is_free_carry,
        options=fsm_options,
    )
    fsm_result = fsm.run(
        q_home=q_start,
        X_WG_pregrasp=X_WG_pregrasp,
        X_WG_drop=X_WG_drop,
    )
    if not fsm_result.success or fsm_result.plan is None:
        print("FSM transitions:", [s.value for s in fsm_result.transition_history])
        raise RuntimeError(f"Manipulation FSM failed: {fsm_result.error_message}")

    plan = fsm_result.plan
    print("FSM transitions:", [s.value for s in fsm_result.transition_history])
    print(f"home->pregrasp path length: {len(plan.path_home_to_pregrasp)}")
    print(f"pregrasp->grasp path length: {len(plan.grasp_plan.path_pregrasp_to_grasp)}")
    print(f"grasp->pregrasp path length: {len(plan.grasp_plan.path_grasp_to_pregrasp)}")
    print(f"pregrasp->drop path length: {len(plan.path_pregrasp_to_drop)}")
    print(f"selected grasp variant: {plan.grasp_plan.selected_variant_index}")
    print(f"selected drop candidate: {plan.drop_candidate_index}")

    lcm_url = scenario.lcm_buses["default"].lcm_url
    lcm = DrakeLcm(lcm_url)

    iiwa_cmd = lcmt_iiwa_command()
    iiwa_cmd.num_joints = 7

    wsg_cmd = lcmt_schunk_wsg_command()
    wsg_cmd.force = 30.0
    # # Gripper's position in mm (0 mm means closed, 1 = 1mm)
    # wsg_cmd.target_position_mm = 0.01

    wsg_closed_mm = 0.001
    wsg_open_mm = 100.0

    # Reset sim state before execution.
    sim_plant.SetPositions(plant_context, iiwa, q_start)
    sim_plant.SetVelocities(plant_context, iiwa, np.zeros(sim_plant.num_velocities(iiwa)))

    # Start recording so we can use play/reset/pause buttons.
    meshcat.StartRecording()

    # Force closed fingers before any arm motion.
    _hold_current_position(
        simulator=simulator,
        q_hold=q_start,
        hold_time=0.25,
        lcm=lcm,
        iiwa_cmd=iiwa_cmd,
        wsg_cmd=wsg_cmd,
        gripper_target_mm=wsg_closed_mm,
        command_dt=0.01,
    )

    # Execute FSM segment plan with explicit gripper events.
    segments = [
        ("home_to_pregrasp", plan.path_home_to_pregrasp, wsg_closed_mm),
        ("pregrasp_to_grasp", plan.grasp_plan.path_pregrasp_to_grasp, wsg_open_mm),
        ("grasp_to_pregrasp", plan.grasp_plan.path_grasp_to_pregrasp, wsg_closed_mm),
        ("pregrasp_to_drop", plan.path_pregrasp_to_drop, wsg_closed_mm),
    ]
    for segment_name, segment_path, gripper_target_mm in segments:
        trajectory, duration = _build_trajectory_from_path(segment_path, vmax=0.35)
        print(f"Executing {segment_name}: {len(segment_path)} waypoints, {duration:.2f}s")
        _publish_and_advance_segment(
            simulator=simulator,
            trajectory=trajectory,
            duration=duration,
            lcm=lcm,
            iiwa_cmd=iiwa_cmd,
            wsg_cmd=wsg_cmd,
            gripper_target_mm=gripper_target_mm,
            command_dt=0.01,
        )

        # Open at pregrasp before the approach-to-grasp segment.
        if segment_name == "home_to_pregrasp":
            _hold_current_position(
                simulator=simulator,
                q_hold=plan.q_pregrasp,
                hold_time=0.35,
                lcm=lcm,
                iiwa_cmd=iiwa_cmd,
                wsg_cmd=wsg_cmd,
                gripper_target_mm=wsg_open_mm,
                command_dt=0.01,
            )

        # Close after reaching grasp waypoint, before retreat.
        if segment_name == "pregrasp_to_grasp":
            _hold_current_position(
                simulator=simulator,
                q_hold=plan.q_grasp,
                hold_time=0.50,
                lcm=lcm,
                iiwa_cmd=iiwa_cmd,
                wsg_cmd=wsg_cmd,
                gripper_target_mm=wsg_closed_mm,
                command_dt=0.01,
            )

    # Open and settle at drop.
    _hold_current_position(
        simulator=simulator,
        q_hold=plan.q_drop,
        hold_time=1.00,
        lcm=lcm,
        iiwa_cmd=iiwa_cmd,
        wsg_cmd=wsg_cmd,
        gripper_target_mm=wsg_open_mm,
        command_dt=0.01,
    )

    meshcat.StopRecording()
    meshcat.PublishRecording()
    meshcat.StaticHtml()
    with open("rrt_connect.html", "w") as f:
        f.write(meshcat.StaticHtml())

    return 0


if __name__ == "__main__":
    main()
