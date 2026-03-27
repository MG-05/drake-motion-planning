import numpy as np


def is_collision_free(
    plant,
    scene_graph,
    root_context,
    iiwa_instance,
    wsg_instance=None,
    q_wsg_instance=None,
    min_clearance=0.005,
    pair_range=0.03,
    ignore_model_instances=None,
    extra_checked_model_instances=None,
):
    """
    Returns a function is_free for a given model instance. We will use SceneGraph
    QueryObject penetrations as a collision test.

    Effectivly, we do a collision check for every q
    """

    # Clone root_context
    check_root = root_context.Clone()
    plant_context = plant.GetMyMutableContextFromRoot(check_root)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(check_root)

    number_iiwa = plant.num_positions(iiwa_instance)

    # include the gripper in collision checks if it is attached
    if wsg_instance is not None:
        number_wsg = plant.num_positions(wsg_instance)

        if q_wsg_instance is None:
            q_wsg_fixed = plant.GetPositions(plant_context, wsg_instance).copy()
        else:
            q_wsg_fixed = np.asarray(q_wsg_instance).copy()

        q_wsg_fixed = np.asarray(q_wsg_fixed).reshape((number_wsg,))
    else:
        q_wsg_fixed = None

    robot_geom_ids = set()
    for inst in [iiwa_instance, wsg_instance]:
        if inst is None:
            continue
        for body_index in plant.GetBodyIndices(inst):
            body = plant.get_body(body_index)
            for gid in plant.GetCollisionGeometriesForBody(body):
                robot_geom_ids.add(gid)

    # Optional payload or other models to include to robot geometry as part of the moving body.
    if extra_checked_model_instances is not None:
        for inst in extra_checked_model_instances:
            if inst is None:
                continue
            for body_index in plant.GetBodyIndices(inst):
                body = plant.get_body(body_index)
                for gid in plant.GetCollisionGeometriesForBody(body):
                    robot_geom_ids.add(gid)

    ignored_geom_ids = set()
    if ignore_model_instances is not None:
        for inst in ignore_model_instances:
            if inst is None:
                continue
            for body_index in plant.GetBodyIndices(inst):
                body = plant.get_body(body_index)
                for gid in plant.GetCollisionGeometriesForBody(body):
                    ignored_geom_ids.add(gid)

    # Rigidly attached payload model that should move with a carrier frame.
    attached_model = {
        "model_instance": None,
        "floating_body": None,
        "carrier_frame": None,
        "X_CB": None,
    }

    def configure_attached_model(model_instance, carrier_frame, X_CB):
        payload_body = None
        for body_index in plant.GetBodyIndices(model_instance):
            body = plant.get_body(body_index)
            if body.is_floating():
                payload_body = body
                break
        if payload_body is None:
            raise RuntimeError(
                f"Model instance {model_instance} has no floating body to attach for carry checking"
            )
        attached_model["model_instance"] = model_instance
        attached_model["floating_body"] = payload_body
        attached_model["carrier_frame"] = carrier_frame
        attached_model["X_CB"] = X_CB

    def clear_attached_model():
        attached_model["model_instance"] = None
        attached_model["floating_body"] = None
        attached_model["carrier_frame"] = None
        attached_model["X_CB"] = None

    def _set_positions(q_iiwa):
        q_iiwa = np.asarray(q_iiwa).reshape((number_iiwa,))
        plant.SetPositions(plant_context, iiwa_instance, q_iiwa)

        if wsg_instance is not None:
            plant.SetPositions(plant_context, wsg_instance, q_wsg_fixed)

        # If configured, move the payload rigidly with the carrier frame.
        if attached_model["floating_body"] is not None:
            X_WC = plant.CalcRelativeTransform(
                plant_context,
                plant.world_frame(),
                attached_model["carrier_frame"],
            )
            X_WB = X_WC @ attached_model["X_CB"]
            plant.SetFreeBodyPose(plant_context, attached_model["floating_body"], X_WB)

    def _robot_env_penetrations(query):
        penetrations = query.ComputePointPairPenetration()
        robot_env_pens = []
        for pen in penetrations:
            # only append collisions between objects a and b such that only object a XOR b is a robot
            # we effectively check for collisions between robot and enviorment only because robot to robot
            # and robot to enviorment was taking far too long.
            a_robot = (pen.id_A in robot_geom_ids)
            b_robot = (pen.id_B in robot_geom_ids)
            if a_robot != b_robot:
                if (pen.id_A in ignored_geom_ids) or (pen.id_B in ignored_geom_ids):
                    continue
                robot_env_pens.append(pen)
        return robot_env_pens

    def _robot_env_min_signed_distance(query):
        if not hasattr(query, "ComputeSignedDistancePairwiseClosestPoints"):
            return None

        pairs = query.ComputeSignedDistancePairwiseClosestPoints(pair_range)
        min_d = float(pair_range)
        found_pair = False
        for p in pairs:
            # similar to collision, only append clearance-collisions between objects a and b such that only
            # object a XOR b is a robot
            # we effectively check for clearance-collisions between robot and enviorment only because robot to robot
            # and robot to enviorment was taking far too long.
            a_robot = (p.id_A in robot_geom_ids)
            b_robot = (p.id_B in robot_geom_ids)
            if a_robot != b_robot:
                if (p.id_A in ignored_geom_ids) or (p.id_B in ignored_geom_ids):
                    continue
                found_pair = True
                if p.distance < min_d:
                    # take note of the smallest robot to enviorment distance
                    min_d = p.distance
        if found_pair:
            return float(min_d)
        # No nearby pairs means the clearance exceeds pair_range.
        return float(pair_range)

    def estimate_clearance(q_iiwa):
        _set_positions(q_iiwa)
        query = scene_graph.get_query_output_port().Eval(scene_graph_context)

        robot_env_pens = _robot_env_penetrations(query)
        if robot_env_pens:
            max_depth = max(float(pen.depth) for pen in robot_env_pens)
            return -max_depth

        min_distance = _robot_env_min_signed_distance(query)
        if min_distance is not None:
            return float(min_distance)
        return float("inf")

    def is_free(q_iiwa):
        _set_positions(q_iiwa)
        query = scene_graph.get_query_output_port().Eval(scene_graph_context)

        # Fast penetration check for robot vs environment only
        robot_env_pens = _robot_env_penetrations(query)

        if min_clearance <= 1e-12:
            if robot_env_pens:
                # fail if penetrates
                return False
            return True

        # Clearance check for robot vs environment only
        min_d = _robot_env_min_signed_distance(query)
        if min_d is None:
            return True
        if min_d < float(min_clearance):
            # clearence violated so reject config
            return False
        return True

    is_free.configure_attached_model = configure_attached_model
    is_free.clear_attached_model = clear_attached_model
    is_free.estimate_clearance = estimate_clearance
    is_free.minimum_clearance = float(min_clearance)
    is_free.pair_range = float(pair_range)

    return is_free
