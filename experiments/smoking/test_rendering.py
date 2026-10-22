import math
import unittest

from rendering import overview_position


class CameraFramingTests(unittest.TestCase):
    def test_fly_and_sources_stay_inside_frame_even_far_from_arena(self):
        sources = [(14, 7), (14, -7)]
        for fly in [(0, 0, 0.2), (300, -200, 1), (-500, 400, 3), (14, 7, 100)]:
            for aspect in [800 / 608, 0.5, 2.0]:
                with self.subTest(fly=fly, aspect=aspect):
                    camera = overview_position(fly, sources, aspect=aspect)
                    for point in [fly, (*sources[0], 0.25), (*sources[1], 0.25)]:
                        depth = camera[2] - point[2]
                        half_height = depth * math.tan(math.radians(45) / 2)
                        self.assertGreater(depth, 0)
                        self.assertLess(abs(point[0] - camera[0]), half_height * aspect)
                        self.assertLess(abs(point[1] - camera[1]), half_height)
                        pixel_y = (1 - (point[1] - camera[1]) / half_height) * 608 / 2
                        self.assertGreater(pixel_y, 130)  # below the HUD
                        self.assertLess(pixel_y, 390)  # above the map


if __name__ == '__main__':
    unittest.main()
