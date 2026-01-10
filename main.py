import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    Parser,
    PiecewisePolynomial,
    Simulator,
    StartMeshcat,
    AddDefaultVisualization,
    StateInterpolatorWithDiscreteDerivative,
    TrajectorySource,
)

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant, scene_graph)

    iiwa_url = "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    iiwa = parser.AddModelsFromUrl(iiwa_url)[0]

    # Weld base to world
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", iiwa))
    plant.Finalize()

    # Meshcat visualization
    meshcat = StartMeshcat()
    AddDefaultVisualization(builder, meshcat=meshcat)

    nq = plant.num_positions(iiwa)

    # Home posture (7 joints).
    q_home = np.array([0, 0.5, 0.32, -1.76, -0.36, 0.83, -0.50])
    q_wave = q_home.copy()
    # move joint 1
    q_wave[0] += 2
    # move joint 2
    q_wave[5] += -2.5

    times = [0.0, 2.0, 4.0, 6.0]
    t_final = times[-1] + 1.0
    # go to final, home, final
    knots = np.vstack([q_home, q_wave, q_home, q_wave]).T
    q_traj = PiecewisePolynomial.FirstOrderHold(times, knots)

    traj_src = builder.AddSystem(TrajectorySource(q_traj))
    # interpolate slowly
    interp = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(nq, 0.00005, suppress_initial_transient=True)
    )
    builder.Connect(traj_src.get_output_port(), interp.get_input_port())

    # inverse dynamics controller with PID
    kp = 60 * np.ones(nq)
    ki = 20 * np.ones(nq)
    kd = 30 * np.ones(nq)
    controller = builder.AddSystem(
        InverseDynamicsController(plant, kp, ki, kd, has_reference_acceleration=False)
    )

    builder.Connect(interp.get_output_port(), controller.get_input_port_desired_state())
    builder.Connect(plant.get_state_output_port(), controller.get_input_port_estimated_state())
    builder.Connect(controller.get_output_port_control(), plant.get_actuation_input_port())

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # Initialize at home
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, iiwa, q_home)

    print("Meshcat URL:", meshcat.web_url())

    # print positions with position checks every dt
    dt = 0.02
    while (simulator.get_context().get_time() < t_final):
        q = plant.GetPositions(plant_context, iiwa)
        # print("q:", q)
        simulator.AdvanceTo(simulator.get_context().get_time() + dt)
        if np.allclose(q, q_wave, atol=1e-4, rtol=1e-2):
            print(f"Reached end State at t = {simulator.get_context().get_time():.2f}")

    input("\nPress enter to quit.")

if __name__ == "__main__":
    main()
