import unittest
from recovery import StuckRecovery


class RecoveryTests(unittest.TestCase):
    def test_stall_runs_reposition_turn_and_resume_then_resets(self):
        recovery = StuckRecovery()
        for i in range(8):
            self.assertEqual(recovery.update(i * .15, (0, 0), True)[0], '')
        self.assertEqual(recovery.update(1.2, (0, 0), True), ('REPOSITION', (.8, .8)))
        self.assertEqual(recovery.update(1.65, (0, 0), True)[0], 'TURN')
        self.assertEqual(recovery.update(2.25, (0, 0), True), ('RESUME', (.8, .8)))
        self.assertEqual(recovery.update(2.7, (1, 0), True)[0], '')
        self.assertEqual(recovery.attempts, 1)

    def test_feeding_and_boundary_cancel_without_false_alarm(self):
        recovery = StuckRecovery()
        for i in range(100):
            self.assertEqual(recovery.update(i * .15, (0, 0), False), ('', None))
        self.assertEqual(recovery.attempts, 0)
        for i in range(9):
            recovery.update(20 + i * .15, (0, 0), True)
        self.assertEqual(recovery.attempts, 1)
        self.assertEqual(recovery.update(21.3, (0, 0), False), ('', None))
        self.assertIsNone(recovery.started)

    def test_moving_and_closed_loops_are_not_stalls(self):
        recovery = StuckRecovery()
        for i in range(100):
            self.assertEqual(recovery.update(i * .15, (i % 4, 0), True)[0], '')
        self.assertEqual(recovery.attempts, 0)

    def test_recovery_never_reverses_leg_frequencies(self):
        recovery = StuckRecovery()
        stages = set()
        for i in range(50):
            state, action = recovery.update(i*.15, (0,0), True)
            if action:
                self.assertTrue(all(0 < drive <= 1 for drive in action))
                stages.add(state)
        self.assertEqual(stages, {'REPOSITION', 'TURN', 'RESUME'})
