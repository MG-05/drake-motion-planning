import numpy as np

def is_collision_free(diagram, plant, scene_graph, root_context, model_instance):
    """
    Returns a function is_free for a given model instance. We will use SceneGraph
    QueryObject penetrations as a collision test.

    Effectivly, we do a collision check for every q
    """

    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(root_context)

    number_q = plant.num_positions(model_instance)

    def is_free(q):
        q = np.asarray(q).reshape((number_q, ))
        plant.SetPositions(plant_context, model_instance, q)

#         evaluate if object is penetrating something or not
        query = scene_graph.get_query_output_port().Eval(scene_graph_context)
        num_penetrations = len(query.ComputePointPairPenetration())

        # return if penetrating or not
        return num_penetrations == 0

    return is_free

