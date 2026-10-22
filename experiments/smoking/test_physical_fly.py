import unittest
import numpy as np
from physical_fly import ExperimentFly
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import HybridTurningController
from scenery import add_scenery


class VisualContactTests(unittest.TestCase):
    def test_walk_through_grass_and_contacts_exclude_visual_objects(self):
        arena=FlatTerrain()
        items=[dict(kind='grass',position=(x,y),scale=1,radius=.3,angle=0)
               for x in range(6) for y in (-.5,0,.5)]
        add_scenery(arena,items)
        fire=arena.root_element.worldbody.add('body',name='fire_test',pos=(3,0,0))
        fire.add('geom',type='sphere',size=(1,),contype=0,conaffinity=0)
        contacts=[f'{leg}{seg}' for leg in ['LF','LM','LH','RF','RM','RH']
                  for seg in ['Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]
        sim=HybridTurningController(fly=ExperimentFly(enable_adhesion=True,contact_sensor_placements=contacts),arena=arena,seed=0)
        try:
            for pair in arena.root_element.contact.get_children('pair'):
                geom=pair.geom2
                self.assertEqual(getattr(geom,'name',str(geom)),'ground')
            for first,second in zip(sim.physics.model.pair_geom1,sim.physics.model.pair_geom2):
                for geom_id in (first,second):
                    name=sim.physics.model.id2name(int(geom_id),'geom')
                    self.assertTrue(name=='ground' or name.startswith(f'{sim.fly.name}/'),name)
            obs,_=sim.reset(0);start=obs['fly'][0][:2].copy()
            for _ in range(4000):obs,*_=sim.step(np.array([.8,.8]))
            self.assertGreater(np.linalg.norm(obs['fly'][0][:2]-start),2)
            self.assertTrue(np.isfinite(obs['fly']).all())
        finally:sim.close()
