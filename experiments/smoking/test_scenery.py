import math
import unittest
from scenery import random_cigarettes, scenery_layout, segment_distance


class LayoutTests(unittest.TestCase):
    def test_seed_spacing_and_connectivity(self):
        for count in (1, 5, 12):
            for seed in (0, 42, 999):
                points = random_cigarettes(seed, count)
                self.assertEqual(points, random_cigarettes(seed, count))
                self.assertEqual(len(points), count)
                for i, p in enumerate(points):
                    self.assertGreaterEqual(math.dist(p, (0, 0)), 8)
                    self.assertGreaterEqual(math.dist(p, (14, -7)), 8)
                    for q in points[:i]:
                        self.assertGreaterEqual(math.dist(p, q), 8)
                reached, pending = [(0, 0)], list(points)
                while pending:
                    connected = [p for p in pending if any(math.dist(p, q) <= 22 for q in reached)]
                    self.assertTrue(connected)
                    reached.extend(connected)
                    pending = [p for p in pending if p not in connected]
        self.assertNotEqual(random_cigarettes(0), random_cigarettes(42))

    def test_scenery_keeps_connections_clear(self):
        sources = random_cigarettes(42)
        items = scenery_layout(42, sources, (0, 0, .2), (14, -7))
        self.assertEqual(items, scenery_layout(42, sources, (0, 0, .2), (14, -7)))
        self.assertEqual(set(i['kind'] for i in items), {'grass', 'tree', 'rock'})
        protected = [(0, 0), *sources, (14, -7)]
        for item in items:
            if item['kind'] == 'grass':
                continue
            for i, a in enumerate(protected):
                for b in protected[i+1:]:
                    self.assertGreaterEqual(segment_distance(item['position'], a, b), 6 + item['radius'])
