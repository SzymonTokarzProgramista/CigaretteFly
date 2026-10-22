"""Explicit behavior controller; this sequence is not learned by the connectome."""
import math


class SmokingCycle:
    def __init__(self, field, fires, pickup_radius=2.5, ignition_radius=2.0, ignition_sec=0.6):
        self.field, self.fires = field, tuple(fires)
        self.pickup_radius, self.ignition_radius = pickup_radius, ignition_radius
        self.ignition_sec = ignition_sec
        self.state, self.carried, self.fire = 'SEARCH_CIGARETTE', None, None
        self.ignition_progress, self.finished = 0.0, 0
        self.last_intake_index = None

    def update(self, position, dt):
        self.last_intake_index = None
        if self.state == 'SEARCH_CIGARETTE':
            index = self.field.contact(position, self.pickup_radius)
            if index is not None:
                self.carried = index
                self.field.excluded.add(index)
                self.state = 'SEARCH_FIRE'
            return 0.0
        if self.state == 'LEAVE_FIRE':
            if all(math.dist(position[:2], p) > self.ignition_radius + 2.0 for p in self.fires):
                self.fire = None
                self.state = 'SEARCH_CIGARETTE'
            return 0.0
        if self.state in ('SEARCH_FIRE', 'IGNITING'):
            radius = self.ignition_radius + (0.6 if self.state == 'IGNITING' else 0.0)
            near = [i for i, p in enumerate(self.fires) if math.dist(position[:2], p) < radius]
            if not near:
                self.fire, self.ignition_progress, self.state = None, 0.0, 'SEARCH_FIRE'
                return 0.0
            if near[0] != self.fire:
                self.ignition_progress = 0.0
            self.fire = near[0]
            self.state = 'IGNITING'
            self.ignition_progress += dt
            if self.ignition_progress + 1e-9 >= self.ignition_sec:
                self.state = 'SMOKING'
            return 0.0
        self.last_intake_index = self.carried
        intake = self.field.supplies[self.carried].consume(dt)
        if not self.field.supplies[self.carried].available:
            self.finished += 1
            self.carried, self.ignition_progress = None, 0.0
            self.state = 'LEAVE_FIRE'
        return intake

    def fire_turn(self, position, heading):
        # The explicitly authorized behavioral controller chooses a fire target.
        target = min(self.fires, key=lambda p: math.dist(position[:2], p))
        dx, dy = target[0]-position[0], target[1]-position[1]
        if self.state == 'LEAVE_FIRE':
            dx, dy = -dx, -dy
            if math.hypot(dx,dy) < 1e-6:
                dx, dy = heading[:2]
        angle = math.atan2(heading[0]*dy-heading[1]*dx, heading[0]*dx+heading[1]*dy)
        return max(-0.32, min(0.32, -0.8*angle))
