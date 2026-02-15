import numpy as np

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Body, ModelInstanceIndex


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Returns v normalized to unit length.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def get_floating_body(plant: MultibodyPlant, model_instance: ModelInstanceIndex) -> Body:
    """
    Returns the floating body in a model instance (isolate the brick)
    """
    for body_idx in plant.GetBodyIndices(model_instance):
        body = plant.get_body(body_idx)
        if body.is_floating():
            return body
    raise RuntimeError(
        f"Model instance {model_instance} has no floating body; "
        "is the brick welded or not free?"
    )


def get_body_pose_in_world(plant: MultibodyPlant, plant_context, body: Body) -> RigidTransform:
    """
    Returns X_WB for a given body B (brick).
    X_WB is the roto-translational matrix defining the block's orientation and rotation in the world (3x4)
    """
    return plant.CalcRelativeTransform(plant_context, plant.world_frame(), body.body_frame())


def get_iiwa_base_position_W(plant: MultibodyPlant, plant_context, iiwa_instance: ModelInstanceIndex) -> np.ndarray:
    """
    Returns the iiwa base origin position in world.
    """
    base_body = plant.GetBodyByName("base", iiwa_instance)
    X_WBase = get_body_pose_in_world(plant, plant_context, base_body)
    return X_WBase.translation()


def make_gripper_rotation_from_approach_and_brick(
    approach_W: np.ndarray,
    R_WB: RotationMatrix,
    world_up_W: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> RotationMatrix:
    """
    Builds a gripper orientation R_WG with:
      - gripper +x pointing along approach_W (attack axis),
      - gripper jaw axis (+y) aligned as much as possible with a brick axis.

    Thought process:
    - We need a *repeatable* end-effector orientation to make IK feasible.
    - Approach axis: x_g = direction we “fly in” toward the brick.
    - Jaw axis: for a parallel jaw gripper, fingers open/close along y_g.
      We try to align y_g with one of the brick’s in-plane axes so the
      gripper is “square” to the brick rather than arbitrarily rotated.
    - Then z_g is forced by right-handedness: z_g = x_g × y_g.
    """
    up = _unit(world_up_W)
    y = np.asarray(approach_W, dtype=float).reshape(3)
    y[2] = 0.0  # keep approach horizontal to avoid diving into the shelf
    y = _unit(y)

    # Candidate jaw axes from the brick frame (brick x/y axes in world)
    R = R_WB.matrix()
    b_x = R[:, 0]
    b_y = R[:, 1]

    def proj_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
        return v - np.dot(v, n) * n

    # Prefer a jaw axis orthogonal to approach, so project onto plane normal to x
    cand1 = proj_to_plane(b_x, y)
    cand2 = proj_to_plane(b_y, y)

    # Pick the better-conditioned candidate (larger norm after projection)
    x = cand1 if np.linalg.norm(cand1) >= np.linalg.norm(cand2) else cand2
    if np.linalg.norm(x) < 1e-6:
        # Fallback if brick axes are degenerate with approach
        x = np.cross(y, up)
    x = _unit(x)

    # Orthonormalize y against x (just in case)
    x = _unit(x - np.dot(x, y) * y)

    # Right-handed z
    z = _unit(np.cross(x, y))

    # Ensure z points generally “up” (not flipped upside down)
    if np.dot(z, up) < 0.0:
        x = -x
        z = -z

    # Recompute y to be exactly orthonormal
    x = _unit(np.cross(y, z))

    R_WG = np.column_stack([x, y, z])
    return RotationMatrix(R_WG)


def compute_pregrasp_pose_for_brick(
    plant: MultibodyPlant,
    plant_context,
    iiwa_instance: ModelInstanceIndex,
    brick_body: Body,
    # 40cm clearance from brick
    fingertip_clearance_m: float = 0.04,
    # Emperically approx distance from wsg::body origin to the fingertip contact line.
    wsg_body_to_fingertips_m: float = 0.14,
) -> RigidTransform:
    """
    Computes a world-frame pregrasp pose X_WG for the *wsg::body* frame.

    Thought process:
    - User asked: “pre-grasp a few cm away from the brick on the axis of attack”.
    - The attack axis is taken as robot-base → brick direction (XY-projected).
    - The pregrasp distance should be measured from *fingertips*, but IK targets
      the *wsg::body* frame. So we back off by:
          (wsg_body_to_fingertips_m + fingertip_clearance_m)
    - Result: fingertips end up ~fingertip_clearance_m away from brick.

    Returns:
    - X_WG (RigidTransform): desired pose of the gripper body frame in world.
    """
    X_WB = get_body_pose_in_world(plant, plant_context, brick_body)
    p_WB = X_WB.translation()
    R_WB = X_WB.rotation()

    p_WBase = get_iiwa_base_position_W(plant, plant_context, iiwa_instance)

    # Attack/approach axis: base -> brick (horizontal)
    approach_W = p_WB - p_WBase
    approach_W[2] = 0.0
    if np.linalg.norm(approach_W) < 1e-6:
        approach_W = np.array([0.0, 1.0, 0.0])

    R_WG = make_gripper_rotation_from_approach_and_brick(approach_W, R_WB)

    # Back off from the brick along the approach axis by (tool length + clearance).
    # +Y is forward for the WSG
    approach_axis_W = R_WG.matrix()[:, 1]

    # x_axis_W = R_WG.matrix()[:, 0]
    keep_distance_from_brick = float(wsg_body_to_fingertips_m + fingertip_clearance_m)
    p_WG = p_WB - keep_distance_from_brick * approach_axis_W

    return RigidTransform(R_WG, p_WG)