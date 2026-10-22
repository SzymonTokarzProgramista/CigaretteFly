"""Finite cigarette supply measured in seconds of nicotine intake."""
from dataclasses import dataclass


@dataclass
class CigaretteSupply:
    capacity_sec: float
    consumed_sec: float = 0.0

    @property
    def remaining_sec(self):
        return max(0.0, self.capacity_sec - self.consumed_sec)

    @property
    def available(self):
        return self.remaining_sec > 1e-9

    @property
    def fraction(self):
        return self.remaining_sec / self.capacity_sec

    def consume(self, seconds):
        used = min(max(seconds, 0.0), self.remaining_sec)
        self.consumed_sec = min(self.capacity_sec, self.consumed_sec + used)
        if not self.available:
            self.consumed_sec = self.capacity_sec
        return used


class CigaretteField:
    """Independent supplies, shared odor identity, and local contact only."""
    def __init__(self, positions, capacity_sec):
        self.excluded = set()
        self.positions = tuple(positions)
        self.supplies = [CigaretteSupply(capacity_sec) for _ in positions]

    def distances(self, position):
        import math
        return [math.hypot(position[0] - p[0], position[1] - p[1]) for p in self.positions]

    def nearest(self, position, available_only=True):
        candidates = [i for i, s in enumerate(self.supplies) if (s.available and i not in self.excluded) or not available_only]
        distances = self.distances(position)
        return min(candidates, key=lambda i: distances[i]) if candidates else None

    def distance(self, position):
        index = self.nearest(position)
        if index is None:
            index = self.nearest(position, available_only=False)
        return self.distances(position)[index]

    def contact(self, position, radius):
        index = self.nearest(position)
        return index if index is not None and self.distances(position)[index] < radius else None

    def odor(self, position, sigma):
        import math
        # Strongest local plume: overlapping sources do not create a false
        # feeding zone far from every object through concentration saturation.
        return max((math.exp(-d*d / (2*sigma*sigma))
                    for i, (d, supply) in enumerate(zip(self.distances(position), self.supplies))
                    if supply.available and i not in self.excluded), default=0.0)
