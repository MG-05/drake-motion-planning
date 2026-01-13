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
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig


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

    # Publish to see the geometry of the enviorment
    diagram.ForcedPublish(simulator.get_context())

    # have scenario.simulation_duration set to +inf so will run the init environment indefinitely
    duration = float(args.duration) if args.duration is not None else float(scenario.simulation_duration)
    print(f"[Sim] Advancing to t = {duration:.3f} s")
    simulator.AdvanceTo(duration)

    return 0


if __name__ == "__main__":
    main()
