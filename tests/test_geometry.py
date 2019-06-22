from unittest import TestCase

from gis.datum import Crs
from gis.geometry import Wkt, Point, Extent, GeometryFrame, Polygon, InteractiveGeometryPlotter


class TestGeometry(TestCase):

    POLYGON = "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))"
    LINESTRING = "LINESTRING(-71.160281 42.258729,-71.160837 42.259113,-71.161144 42.25932)"
    POINT = "POINT(-71 42)"
    shape_path = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\shapes\\domy.shp"

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

        frame = GeometryFrame.from_file(self.shape_path).show(2, True)

    def test_extent_to_wkt(self):
        extent = Extent()
        self.assertEqual(extent.to_wkt(), "POLYGON((0.0 0.0, 0.0 1.0, 1.0 1.0, 1.0 0.0, 0.0 0.0))")

    def test_point(self):
        p = Point(21.0, 52.0)

    def test_polygon(self):
        coordinates = [[110.0, 110.0], [110.0, 120.0], [120.0, 120.0], [120.0, 110.0], [110.0, 110.0]]
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(coordinates)

    def test_polygon_from_wkt(self):
        poly = Polygon.from_wkt("POLYGON((0.0 0.0, 0.0 1.0, 1.0 1.0, 1.0 0.0, 0.0 0.0))", Crs("local"))
        self.assertEqual(poly.area, 1.0)

    def test_interactive_plot(self):
        gdf = GeometryFrame.from_file(self.shape_path)
        InteractiveGeometryPlotter(gdf).plot()

    def test_gdf_interactive_plot(self):
        gdf = GeometryFrame.from_file(self.shape_path)
        gdf.plot(interactive=True)


