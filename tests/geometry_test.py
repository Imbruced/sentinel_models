import unittest
from gis.geometry import Origin
from gis.log_lib import logger


class GeometryOriginTest(unittest.TestCase):

    def test_origin_constructor(self):
        origin = Origin(x=1, y=2)
        self.assertEqual(origin.x, 1)
        self.assertEqual(origin.y, 2)
        logger.info(f"Origin has proper constructor")

    def test_str_method(self):
        origin = Origin(x=1, y=2)
        self.assertEqual(str(origin), "Point(1 2)")
