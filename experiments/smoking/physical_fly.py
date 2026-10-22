"""FlyGym contact setup restricted to real walkable/collidable terrain."""
from flygym import Fly


class ExperimentFly(Fly):
    def __init__(self, *args, solid_scenery=False, **kwargs):
        self.solid_scenery = solid_scenery
        super().__init__(*args, **kwargs)

    def init_floor_contacts(self, arena):
        super().init_floor_contacts(arena)
        allowed = set()
        for geom in arena.root_element.find_all('geom'):
            parent_name = getattr(geom.parent, 'name', '') or ''
            if geom.name == 'ground' or (self.solid_scenery and
                    parent_name.startswith(('tree_', 'rock_')) and geom.contype == 1):
                allowed.add(geom.full_identifier)
                allowed.add(geom.name)
        for pair in list(arena.root_element.contact.get_children('pair')):
            target = pair.geom2
            name = getattr(target, 'full_identifier', str(target))
            if name not in allowed:
                pair.remove()
        # Keep FlyGym's own bookkeeping consistent with the MJCF contact list.
        self._floor_contacts = {k:v for k,v in self._floor_contacts.items() if v.parent is not None}
