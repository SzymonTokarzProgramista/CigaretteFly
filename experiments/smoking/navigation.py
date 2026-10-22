"""Sensory motor adapter: learned valence selects attraction, antennae select direction."""


def learned_odor_turn(left_a, right_a, left_b, right_b, relative_valence,
                      feeding_valence=0.8, gain=8.0):
    # No object coordinates, identities, or waypoint sequence enter this adapter.
    # Use the same learned/naive threshold as the feeding reflex. A weak
    # but established association must still yield a measurable turn.
    preference = 1.0 if relative_valence < feeding_valence else -1.0
    gradient_a = (left_a - right_a) / max(left_a + right_a, 0.02)
    gradient_b = (left_b - right_b) / max(left_b + right_b, 0.02)
    strength_a = max(left_a, right_a)
    strength_b = max(left_b, right_b)
    # Increasing left leg drive turns right: attraction needs a negative delta.
    turn = gain * (-preference * gradient_a * strength_a + 0.25 * gradient_b * strength_b)
    return max(-0.32, min(0.32, turn))


def boundary_turn(position, heading, bounds, margin=10.0):
    """Wall sensing reflex: returns None when no wall approach is detected."""
    xmin, xmax, ymin, ymax = bounds
    inward_x = (1 if position[0] < xmin + margin else 0) - (1 if position[0] > xmax - margin else 0)
    inward_y = (1 if position[1] < ymin + margin else 0) - (1 if position[1] > ymax - margin else 0)
    if not (inward_x or inward_y):
        return None
    if inward_x * heading[0] + inward_y * heading[1] > 0.5:
        return None
    cross = heading[0] * inward_y - heading[1] * inward_x
    return -0.32 if cross > 0 else 0.32


def odor_compass_turn(odor_a, odor_b, relative_valence, threshold=0.8):
    """Four local plume samples: forward, backward, left, right.

    Explicit sensory adapter, not a connectome motor circuit. No target
    position or source identity is supplied. Rear samples allow a U-turn.
    """
    import math
    preference = 1.0 if relative_valence < threshold else -1.0
    forward = preference*(odor_a[0]-odor_a[1]) - .25*(odor_b[0]-odor_b[1])
    lateral = preference*(odor_a[2]-odor_a[3]) - .25*(odor_b[2]-odor_b[3])
    if math.hypot(forward,lateral) < 1e-8:
        return 0.24  # bounded exploration in the absence of a directional cue
    angle = math.atan2(lateral,forward)
    return max(-.32,min(.32,-.8*angle))
