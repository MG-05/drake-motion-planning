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

from src.planning.IK import solve_iiwa_ik_for_gripper_pose
from src.planning.collision import is_collision_free
from src.planning.rrt import rrt_plan


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

    iiwa = sim_plant.GetModelInstanceByName("iiwa")

    # Start Config
    q_start = sim_plant.GetPositions(plant_context, iiwa).copy()

    # Forward Kinematics for Goal Config
    # q_goal = q_start.copy()
    # q_goal[0] = q_goal[0] + 1.8
    # q_goal[1] = q_goal[1] - 0.4
    # q_goal[2] = q_goal[2] - 1.4

    # Inverse Kinematics for Goal Config
    wsg = sim_plant.GetModelInstanceByName("wsg")
    # xyz
    # top shelf = np.array([-0.1, -0.05, 0.63])
    # middle shelf = np.array([-0.1, -0.05, 0.39])
    # bottom shelf = np.array([-0.1, -0.05, 0.10])
    # outside = np.array([-0.30, -0.06, 0.44])
    end_effector_pos_desired = np.array([-0.1, -0.05, 0.10])
    # roll pitch yaw
    end_effector_rot_desired = RollPitchYaw(0.0, np.pi/4, 0.0).ToRotationMatrix()
    # Transform Matrix
    transform_desired = RigidTransform(end_effector_rot_desired, end_effector_pos_desired)

    q_goal = solve_iiwa_ik_for_gripper_pose(
        plant=sim_plant,
        plant_context_current=plant_context,
        iiwa_instance=iiwa,
        wsg_instance=wsg,
        desired_end_effector=transform_desired,
        q_iiwa_seed=q_start,
        position_tol=0.05,
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

    # Do an inital check for collisions for the start and final configs
    is_free = is_collision_free(diagram, sim_plant, scene_graph, root_context, iiwa)

    print(f"Is the start config collision free? {is_free(q_start)}")
    print(f"Is the goal config collision free? {is_free(q_goal)}")

    # Plan with RRT
    path = rrt_plan(
        q_start=q_start,
        q_goal=q_goal,
        is_free=is_free,
        joints_lower_limits=joints_lower_limits,
        joints_upper_limits=joints_upper_limits,
        step_size=0.1,
        goal_sample_rate=0.2,
        max_iters=20000,
        edge_resolution=0.02,
        goal_tolerance=0.15
    )

    print(f"The RRT determined path length is {len(path)}")

    # Play the trajectory
    dt = 0.1
    times = np.linspace(0.0, dt*(len(path) - 1), len(path))
    knots = np.array(path).T
    trajectory = PiecewisePolynomial.FirstOrderHold(times, knots)

    # Cloned Context for visualization and to play/pause/reset
    visualize_context = root_context.Clone()
    visualize_plant_context = sim_plant.GetMyMutableContextFromRoot(visualize_context)

    diagram.ForcedPublish(visualize_context)

    try:
        meshcat.DeleteRecording()
    except Exception:
        pass
    meshcat.StartRecording()

    T = times[-1]
    t = 0.0

    while t <= T:
        visualize_context.SetTime(t)
        q = trajectory.value(t).ravel()
        sim_plant.SetPositions(visualize_plant_context, iiwa, q)
        diagram.ForcedPublish(visualize_context)
        # playback rate
        t = t + 0.002

    meshcat.PublishRecording()

    # Publish to see the geometry of the enviorment
    diagram.ForcedPublish(simulator.get_context())

    # have scenario.simulation_duration set to +inf so will run the init environment indefinitely
    duration = float(args.duration) if args.duration is not None else float(scenario.simulation_duration)
    print(f"[Sim] Advancing to t = {duration:.3f} s")
    simulator.AdvanceTo(duration)

    return 0


if __name__ == "__main__":
    main()
