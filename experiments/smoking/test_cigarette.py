import unittest
from cigarette import CigaretteSupply


class SupplyTests(unittest.TestCase):
    def test_partial_contact_and_exhaustion(self):
        supply = CigaretteSupply(0.25)
        self.assertEqual(supply.consume(0.1), 0.1)
        self.assertTrue(supply.available)
        self.assertAlmostEqual(supply.fraction, 0.6)
        self.assertAlmostEqual(supply.consume(1), 0.15)
        self.assertFalse(supply.available)
        self.assertEqual(supply.fraction, 0)
        self.assertEqual(supply.consume(1), 0)

    def test_many_small_steps_cap_intake(self):
        supply = CigaretteSupply(0.2)
        used = sum(supply.consume(0.0001) for _ in range(3000))
        self.assertAlmostEqual(used, 0.2)
        self.assertEqual(supply.remaining_sec, 0)


class FieldTests(unittest.TestCase):
    def test_independent_supplies_and_nearest_available(self):
        from cigarette import CigaretteField
        field = CigaretteField(((0, 0), (20, 0)), 0.2)
        self.assertEqual(field.contact((0, 0), 2.5), 0)
        field.supplies[0].consume(1)
        self.assertIsNone(field.contact((0, 0), 2.5))
        self.assertEqual(field.nearest((0, 0)), 1)
        self.assertEqual(field.supplies[1].remaining_sec, 0.2)
        self.assertLess(field.odor((0, 0), 9), 0.1)
        self.assertEqual(field.odor((20, 0), 9), 1)
        field.supplies[1].consume(1)
        self.assertEqual(field.odor((20, 0), 9), 0)
        self.assertIsNone(field.contact((20, 0), 2.5))

    def test_contact_consumes_only_one_even_when_overlapping(self):
        from cigarette import CigaretteField
        field = CigaretteField(((0, 0), (1, 0)), 5)
        index = field.contact((0, 0), 2.5)
        field.supplies[index].consume(0.1)
        self.assertEqual(field.supplies[1].remaining_sec, 5)
