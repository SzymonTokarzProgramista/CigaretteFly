"""Seeded arena layouts and lightweight MuJoCo scenery, in millimetres."""
import math
import random


def random_cigarettes(seed, count=5, spawn=(0, 0), control=(14, -7)):
    rng = random.Random(seed)
    for _ in range(2000):
        points = []
        for _ in range(2000):
            point = (rng.uniform(7, 65), rng.uniform(-22, 22))
            if all(math.dist(point, p) >= 8 for p in [spawn, control, *points]):
                points.append(point)
                if len(points) == count:
                    break
        # Connected odor neighborhoods; placement does not prescribe a route.
        reached = [spawn]
        pending = points.copy()
        while pending:
            connected = [p for p in pending if any(math.dist(p, q) <= 22 for q in reached)]
            if not connected:
                break
            reached.extend(connected)
            pending = [p for p in pending if p not in connected]
        if len(points) == count and not pending:
            return tuple(points)
    raise ValueError('Cannot generate a connected cigarette layout')


def segment_distance(point, start, end):
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = dx*dx + dy*dy
    t = max(0, min(1, ((point[0]-start[0])*dx + (point[1]-start[1])*dy) / length)) if length else 0
    return math.hypot(point[0]-start[0]-t*dx, point[1]-start[1]-t*dy)


def scenery_layout(seed, sources, spawn, control):
    rng = random.Random(seed)
    protected = [spawn[:2], *sources, control]
    # Keep direct connections clear so decor does not require a new obstacle policy.
    links = [(a, b) for i, a in enumerate(protected) for b in protected[i+1:]]
    bounds = (min(p[0] for p in protected)-12, max(p[0] for p in protected)+12,
              min(p[1] for p in protected)-15, max(p[1] for p in protected)+15)
    items = []
    for kind, count, radius in [('tree', 14, 3.5), ('rock', 22, 1.4), ('grass', 180, 0.3)]:
        for _ in range(count):
            for _ in range(1000):
                point = (rng.uniform(*bounds[:2]), rng.uniform(*bounds[2:]))
                clearance = 6 + radius if kind != 'grass' else 3
                if any(math.dist(point, p) < clearance for p in protected):
                    continue
                if kind != 'grass' and any(segment_distance(point, a, b) < clearance for a, b in links):
                    continue
                if any(math.dist(point, item['position']) < radius + item['radius'] + 1
                       for item in items if item['kind'] != 'grass'):
                    continue
                items.append(dict(kind=kind, position=point, radius=radius,
                                  scale=rng.uniform(0.7, 1.2), angle=rng.uniform(0, math.pi)))
                break
    return items


def add_scenery(arena, items, solid=False):
    # Mutate FlyGym's own ground material, preserving ground contact physics.
    for material in arena.root_element.find_all('material'):
        material.texture = None
        material.rgba = (0.28, 0.40, 0.19, 1)
    for i, item in enumerate(items):
        kind, p, scale = item['kind'], item['position'], item['scale']
        body = arena.root_element.worldbody.add('body', name=f'{kind}_{i}', pos=(*p, 0))
        if kind == 'tree':
            body.add('geom', type='cylinder', pos=(0, 0, 2.5*scale), size=(0.5*scale, 2.5*scale),
                     rgba=(0.34, 0.21, 0.10, 1), contype=int(solid), conaffinity=int(solid))
            for z, radius in [(5.5, 2.3), (7, 1.8), (8.1, 1.1)]:
                body.add('geom', type='ellipsoid', pos=(0, 0, z*scale),
                         size=(radius*scale, radius*scale, 1.5*scale),
                         rgba=(0.16, 0.36+0.02*z, 0.12, 1), contype=0, conaffinity=0)
        elif kind == 'rock':
            body.add('geom', type='ellipsoid', pos=(0, 0, 0.45*scale),
                     size=(1.3*scale, 0.9*scale, 0.65*scale), euler=(0, 0, item['angle']),
                     rgba=(0.43, 0.45, 0.39, 1), contype=int(solid), conaffinity=int(solid))
        else:
            for offset in (-0.15, 0.15):
                body.add('geom', type='capsule', fromto=(offset, 0, 0.02, offset+0.1, 0.1, 0.6*scale),
                         size=(0.035,), rgba=(0.27, 0.49, 0.13, 1), contype=0, conaffinity=0)
