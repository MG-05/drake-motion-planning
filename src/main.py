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
from pydrake.geometry import Meshcat, SceneGraphConfig, CollisionFilterDeclaration, GeometrySet
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
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

# LCM imports
from drake import lcmt_iiwa_command, lcmt_schunk_wsg_command
from pydrake.lcm import DrakeLcm

from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.collision import is_collision_free
from src.planning.rrt import rrt_plan
from src.planning.rrt_connect import rrt_connect_plan
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
    if sim_plant.HasModelInstanceNamed("foam_brick"):
        brick_instance = sim_plant.GetModelInstanceByName("foam_brick")
        brick_body = None
        for body_index in sim_plant.GetBodyIndices(brick_instance):
            body = sim_plant.get_body(body_index)
            if body.is_floating():
                brick_body = body
                break

        if brick_body is None:
            print("Warning: foam_brick has no floating body; skipping free-body pose.")
        else:
            # X_SB = RigidTransform(
            #     RollPitchYaw(0.0, 0.0, 0.0).ToRotationMatrix(),
            #     [0.05, 0.0, 0.20],
            # )
            # X_WB = None
            # if sim_plant.HasModelInstanceNamed("shelves"):
            #     shelves_instance = sim_plant.GetModelInstanceByName("shelves")
            #     try:
            #         shelves_body = sim_plant.GetBodyByName("shelves_body", shelves_instance)
            #         X_WS = sim_plant.CalcRelativeTransform(
            #             plant_context,
            #             sim_plant.world_frame(),
            #             shelves_body.body_frame(),
            #         )
            #         X_WB = X_WS @ X_SB
            #     except Exception:
            #         X_WB = None
            # if X_WB is None:
            #     X_WB = RigidTransform(
            #         RollPitchYaw(0.0, 0.0, math.radians(-90.0)).ToRotationMatrix(),
            #         [0.0, 0.15, 0.5995],
            #     )
            # for bi in sim_plant.GetBodyIndices(brick_instance):
            #     print(sim_plant.get_body(bi).name())

            # Lower shelf -> [0.0, 0.15, 0.030],
            # Middle Shelf
            X_WB = RigidTransform(
                RollPitchYaw(0.0, 0.0, math.radians(-90.0)).ToRotationMatrix(),
                [0.0, 0.15, 0.40],
            )
            sim_plant.SetFreeBodyPose(plant_context, brick_body, X_WB)

    iiwa = sim_plant.GetModelInstanceByName("iiwa")

    # Start Config
    q_start = sim_plant.GetPositions(plant_context, iiwa).copy()

    # Forward Kinematics for Goal Config
    # q_goal = q_start.copy()
    # q_goal[0] = q_goal[0] + 1.8
    # q_goal[1] = q_goal[1] - 0.4
    # q_goal[2] = q_goal[2] - 1.4

    # Inverse Kinematics for Goal Config with collision check on final claw position
    wsg = sim_plant.GetModelInstanceByName("wsg")
    q_wsg_plan = sim_plant.GetPositions(plant_context, wsg).copy()


    is_free = is_collision_free(
        diagram=diagram,
        plant=sim_plant,
        scene_graph=scene_graph,
        root_context=root_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        q_wsg_instance=q_wsg_plan,
        min_clearance=0.01,
        pair_range=0.05,
    )

    # -------------------------
    # Manually define goal end-effector pos + rot
    # -------------------------
    # top shelf = np.array([-0.20, 0.23, 0.63])
    # middle shelf = np.array([0, 0.17, 0.39])
    # bottom shelf = np.array([0.10, 0.10, 0.23])
    # outside = np.array([-0.30, -0.06, 0.44])
    end_effector_pos_desired =  np.array([0, 0.17, 0.39])
    # roll pitch yaw
    end_effector_rot_desired = RollPitchYaw(0.0, 0.0, 0.0).ToRotationMatrix()

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

    end_effector_pos_desired = X_WG_pregrasp.translation()
    end_effector_rot_desired = X_WG_pregrasp.rotation()
    print("\n[Pregrasp] X_WG_pregrasp =", X_WG_pregrasp)

    # Transform Matrix
    transform_desired = RigidTransform(end_effector_rot_desired, end_effector_pos_desired)

    q_goal = solve_iiwa_ik_for_gripper_pose(
        plant=sim_plant,
        root_context_current=root_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        desired_end_effector=transform_desired,
        q_iiwa_seed=q_start,
        position_tol=0.005,
        theta_tol=0.05,
    )

    # Determine Joint Limits for iiwa
    joint_names = [f"iiwa_joint_{i}" for i in range(1, 8)]
    joints_lower_limits = []
    joints_upper_limits = []

    for name in joint_names:
        joint = sim_plant.GetJointByName(name, iiwa)
        joints_lower_limits.append(joint.position_lower_limits()[0])
        joints_upper_limits.append(joint.position_upper_limits()[0])

    joints_lower_limits = np.asarray(joints_lower_limits)
    joints_upper_limits = np.asarray(joints_upper_limits)

    print(f"Is the start config collision free? {is_free(q_start)}")
    print(f"Is the goal config collision free? {is_free(q_goal)}")

    # Plan with RRT-Connect
    path = rrt_connect_plan(
        q_start=q_start,
        q_goal=q_goal,
        is_free=is_free,
        joints_lower_limits=joints_lower_limits,
        joints_upper_limits=joints_upper_limits,
        step_size=0.12,
        goal_sample_rate=0.20,
        max_iters=50000,
        edge_resolution=0.05,
    )


    print(f"The RRT determined path length is {len(path)}")

    # Play the trajectory via velocity based times
    vmax = 0.35  # rad/s
    times = [0.0]
    for i in range(len(path) - 1):
        dq = np.abs(np.array(path[i + 1]) - np.array(path[i]))
        seg_T = max(0.05, float(np.max(dq) / vmax))
        times.append(times[-1] + seg_T)

    times = np.array(times)
    knots = np.array(path).T
    trajectory = PiecewisePolynomial.FirstOrderHold(times, knots)

    lcm_url = scenario.lcm_buses["default"].lcm_url
    lcm = DrakeLcm(lcm_url)

    iiwa_cmd = lcmt_iiwa_command()
    iiwa_cmd.num_joints = 7

    wsg_cmd = lcmt_schunk_wsg_command()
    wsg_cmd.force = 20.0
    # # Gripper's position in mm (0 mm means closed, 1 = 1mm)
    # wsg_cmd.target_position_mm = 0.01

    wsg_closed_mm = 0.001
    wsg_open_mm = 100.0

    # Reset sim state to the known start before executing
    sim_plant.SetPositions(plant_context, iiwa, q_start)
    sim_plant.SetVelocities(plant_context, iiwa, np.zeros(sim_plant.num_velocities(iiwa)))

    # Publish Driver
    t0 = 0.0
    iiwa_cmd.joint_position = trajectory.value(t0).ravel()
    lcm.Publish(channel="IIWA_COMMAND", buffer=iiwa_cmd.encode())
    lcm.Publish(channel="SCHUNK_WSG_COMMAND", buffer=wsg_cmd.encode())

    # advance simulator while pushing commands
    command_dt = 0.01
    T = float(times[-1])

    # Start Recording so we can use play/reset/pause buttons
    meshcat.StartRecording()

    # run simulator for time T
    t = 0.0
    while t < T:
        t_next = min(t + command_dt, T)

        q_des = trajectory.value(t_next).ravel()
        iiwa_cmd.joint_position = q_des
        wsg_cmd.target_position_mm = wsg_closed_mm

        lcm.Publish(channel="IIWA_COMMAND", buffer=iiwa_cmd.encode())
        lcm.Publish(channel="SCHUNK_WSG_COMMAND", buffer=wsg_cmd.encode())

        simulator.AdvanceTo(t_next)
        t = t_next

    # Hold final command for a bit (1 sec.)
    hold_time = 1.0
    t_hold_end = T + hold_time
    while t < t_hold_end:
        t_next = min(t + command_dt, t_hold_end)

        iiwa_cmd.joint_position = trajectory.value(T).ravel()
        wsg_cmd.target_position_mm = wsg_open_mm

        lcm.Publish(channel="IIWA_COMMAND", buffer=iiwa_cmd.encode())
        lcm.Publish(channel="SCHUNK_WSG_COMMAND", buffer=wsg_cmd.encode())

        simulator.AdvanceTo(t_next)
        t = t_next

    meshcat.StopRecording()
    meshcat.PublishRecording()
    meshcat.StaticHtml()
    with open("rrt_connect.html", "w") as f:
        f.write(meshcat.StaticHtml())

    return 0


if __name__ == "__main__":
    main()
