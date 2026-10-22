"""Camera framing independent of the fly's behavioral controller."""
import math


def overview_position(fly_position, sources, aspect=800 / 608, fovy=45, margin=6):
    points = [fly_position, *sources]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    half_width = (max(xs) - min(xs)) / 2 + margin
    half_height = (max(ys) - min(ys)) / 2 + margin
    tangent = math.tan(math.radians(fovy) / 2)
    # Reserve the HUD above and map below the actual scene. Merely fitting
    # the full viewport can hide a distant fly behind those panels.
    height = 2.7 * max(half_height / tangent, half_width / (aspect * tangent))
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2 - 0.15 * height * tangent,
            max(float(fly_position[2]), 1.0) + height)
