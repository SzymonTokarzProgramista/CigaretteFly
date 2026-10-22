import unittest
from cigarette import CigaretteField
from smoking_cycle import SmokingCycle


class CycleTests(unittest.TestCase):
    def test_no_intake_until_ignition_and_remaining_supply_preserved(self):
        field = CigaretteField(((0,0),(20,0)), .2)
        cycle = SmokingCycle(field, ((10,0),), ignition_sec=.3)
        self.assertEqual(cycle.update((0,0),.1),0)
        self.assertEqual(cycle.state,'SEARCH_FIRE')
        self.assertIsNone(field.contact((0,0),2.5))
        for _ in range(10): self.assertEqual(cycle.update((5,0),.1),0)
        self.assertEqual(field.supplies[0].remaining_sec,.2)
        for _ in range(3): self.assertEqual(cycle.update((10,0),.1),0)
        self.assertEqual(cycle.state,'SMOKING')
        self.assertEqual(cycle.update((10,0),.1),.1)
        self.assertEqual(cycle.update((10,0),.1),.1)
        self.assertEqual(cycle.finished,1)
        self.assertEqual(cycle.state,'LEAVE_FIRE')
        cycle.update((15,0),.1)
        self.assertEqual(cycle.state,'SEARCH_CIGARETTE')
        self.assertIsNone(cycle.carried)
        self.assertEqual(field.supplies[1].remaining_sec,.2)

    def test_departure_and_changing_fire_reset_ignition(self):
        cycle=SmokingCycle(CigaretteField(((0,0),),5),((10,0),(20,0)))
        cycle.update((0,0),.1);cycle.update((10,0),.2)
        cycle.update((5,0),.1)
        self.assertEqual(cycle.ignition_progress,0)
        cycle.update((10,0),.2);cycle.update((20,0),.1)
        self.assertAlmostEqual(cycle.ignition_progress,.1)
        self.assertEqual(cycle.state,'IGNITING')

    def test_fire_steering_and_multiple_sources(self):
        cycle=SmokingCycle(CigaretteField(((0,0),),5),((5,5),(100,0)))
        self.assertLess(cycle.fire_turn((0,0),(1,0)),0)
        self.assertGreater(cycle.fire_turn((0,0),(0,1)),0)

    def test_small_body_motion_does_not_restart_ignition(self):
        cycle=SmokingCycle(CigaretteField(((0,0),),5),((4,0),))
        cycle.update((0,0),.1)
        cycle.update((2.1,0),.3)
        cycle.update((1.9,0),.3)
        self.assertEqual(cycle.state,'SMOKING')

    def test_leave_fire_keeps_walking_and_blocks_immediate_pickup(self):
        field=CigaretteField(((0,0),(10,1)), .1)
        cycle=SmokingCycle(field, ((10,0),), ignition_sec=.1)
        cycle.update((0,0),.1)
        cycle.update((10,0),.1)
        cycle.update((10,0),.1)
        self.assertEqual(cycle.state,'LEAVE_FIRE')
        self.assertIsNone(cycle.carried)
        for _ in range(10):
            self.assertEqual(cycle.update((10,1),.1),0)
        self.assertEqual(cycle.state,'LEAVE_FIRE')
        self.assertEqual(field.supplies[1].remaining_sec,.1)
        self.assertLess(cycle.fire_turn((10,1),(1,0)), 0)
        cycle.update((10,5),.1)
        self.assertEqual(cycle.state,'SEARCH_CIGARETTE')
