import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from config import Config
from smoking_experiment import episode


class FailureSaveTests(unittest.TestCase):
    def test_physics_exception_saves_completed_rows_and_weights_then_reraises(self):
        from flygym.examples.locomotion import HybridTurningController
        brain = SimpleNamespace(
            syn=SimpleNamespace(w=np.ones(1)), plastic_pos=np.array([0]),
            A=[0], B=[1], g={k:np.array([i]) for k,i in [('pam',2),('kc',3),('mbon',4)]},
            br={'volt':1}, mbon_reference={'A':100,'B':100}, navigation_valence=.1,
            _episode=lambda *args:np.array([1,1,0,1,1]),
        )
        from dataclasses import replace
        config=replace(Config(), decisions=2, scenery=False, ignition_cycle=False)
        original=HybridTurningController.step
        calls=0
        def injected(controller, action):
            nonlocal calls
            calls+=1
            if calls==1501:
                raise RuntimeError('injected physics failure')
            return original(controller,action)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(HybridTurningController,'step',injected):
                with self.assertRaisesRegex(RuntimeError,'injected physics failure'):
                    episode(brain,config,'A','LIVE',0,out,True,0)
            report=json.loads((out/'A_LIVE_0_partial.json').read_text())
            self.assertEqual(report['completed_decisions'],1)
            self.assertEqual(len((out/'A_LIVE_0_partial.csv').read_text().splitlines()),2)
            self.assertGreater((out/'A_LIVE_0_partial.mp4').stat().st_size,0)
            self.assertTrue(np.array_equal(np.load(out/'A_LIVE_0_partial_weights.npy'),brain.syn.w))
