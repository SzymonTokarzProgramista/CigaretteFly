import unittest
from navigation import learned_odor_turn


class NavigationTests(unittest.TestCase):
    def test_learned_attraction_and_naive_avoidance(self):
        self.assertLess(learned_odor_turn(0.8, 0.6, 0, 0, 0.2), 0)
        self.assertGreater(learned_odor_turn(0.8, 0.6, 0, 0, 1.2), 0)

    def test_reflection_reverses_turn(self):
        a = learned_odor_turn(0.8, 0.6, 0.2, 0.1, 0.2)
        b = learned_odor_turn(0.6, 0.8, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(a, -b)

    def test_depressed_mbon_preserves_sensory_direction(self):
        self.assertLess(learned_odor_turn(0.8, 0.6, 0, 0, 0), 0)
        self.assertEqual(learned_odor_turn(0, 0, 0, 0, 0), 0)


class BoundaryTests(unittest.TestCase):
    def test_no_wall_intervention_in_interior_or_when_heading_inward(self):
        from navigation import boundary_turn
        bounds = (-20, 80, -40, 40)
        self.assertIsNone(boundary_turn((20, 0), (1, 0), bounds))
        self.assertIsNone(boundary_turn((78, 0), (-1, 0), bounds))

    def test_outward_motion_gets_turn_and_reflection_reverses_it(self):
        from navigation import boundary_turn
        bounds = (-20, 80, -40, 40)
        right = boundary_turn((78, 0), (0.9, 0.1), bounds)
        reflected = boundary_turn((78, 0), (0.9, -0.1), bounds)
        self.assertEqual(right, -reflected)
        self.assertIsNotNone(boundary_turn((78, 0), (1, 0), bounds))


class CompassTests(unittest.TestCase):
    def test_source_behind_requires_turn_even_with_balanced_left_right(self):
        from navigation import odor_compass_turn
        self.assertEqual(abs(odor_compass_turn((.2,.8,.4,.4),(0,0,0,0),.1)),.32)

    def test_strength_does_not_erase_direction(self):
        from navigation import odor_compass_turn
        a=odor_compass_turn((.8,.2,.7,.3),(0,0,0,0),.1)
        b=odor_compass_turn((.0008,.0002,.0007,.0003),(0,0,0,0),.1)
        self.assertEqual(a,b)
        self.assertLess(a,0)
