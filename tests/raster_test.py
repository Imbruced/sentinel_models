import unittest
from gis.raster import Raster
from geom.data_types import Extent
from geom.data_types import Point
from gis.raster import RasterShape

pointa = Point(0, 0)
pointb = Point(100, 100)

extent = Extent(pointa, pointb)
raster_shape = RasterShape(extent, 20)


class RasterTest(unittest.TestCase):
    test_cases = [

    ]
    raster = Raster(raster_shape)

    def test_raster_shape(self):
        self.assertEqual(self.raster.array.shape, (100, 100))

