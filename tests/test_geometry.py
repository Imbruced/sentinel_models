from unittest import TestCase

from gis.geometry import Wkt, Point, Extent, GeometryFrame


class TestGeometry(TestCase):

    POLYGON = "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))"
    LINESTRING = "LINESTRING(-71.160281 42.258729,-71.160837 42.259113,-71.161144 42.25932)"
    POINT = "POINT(-71 42)"

    def test_wkt_instance(self):
        wkt = Wkt("Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))")

    def test_split_coordinates(self):
        wkt1 = Wkt(self.POLYGON)
        wkt2 = Wkt(self.POINT)
        wkt3 = Wkt(self.LINESTRING)

        self.assertEqual(
            wkt1.coordinates,
            [
                Point(x[0], x[1]) for x in
                [[110.0, 110.0], [110.0, 120.0], [120.0, 120.0], [120.0, 110.0], [110.0, 110.0]]
             ]
        )
        self.assertEqual(
            wkt2.coordinates,
            [Point(-71.0, 42.0)]
        )

        self.assertEqual(
            wkt3.coordinates,
            [
                Point(x[0], x[1]) for x in
                [[-71.160281, 42.258729], [-71.160837, 42.259113], [-71.161144, 42.25932]]
            ]
        )

    def test_extent_wkt(self):
        wkt1 = Wkt(self.POLYGON)
        self.assertEqual(
            wkt1.extent,
            Extent(Point(x=110.0, y=110.0), Point(x=120.0, y=120.0))
        )

    def test_showing_geoframe_like_spark(self):
        shape_path = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\shapes\\domy.shp"
        frame = GeometryFrame.from_file(shape_path).show(2, True)