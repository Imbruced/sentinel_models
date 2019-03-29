from abc import ABC
import geopandas as gpd
import os
from gis.exceptions import GeometryCollectionError
from gis.exceptions import GeometryTypeError
scriptDirectory = os.path.dirname(os.path.realpath(__file__))
from gis.descriptors import NumberType
import attr
from gis.geometry_operations import count_delta
from gis.validators import IsNumeric

@attr.s
class GeometryFrame(ABC):

    frame = attr.ib()
    geometry_column = attr.ib()
    crs = attr.ib(default=None)

    def __attr__post_init__(self):
        self.type = self.__class__.__name__

    def to_wkt(self):
        """
        Based on geometry column this method returns wkt representation of it, it does not modify
        passed instance, it create new one

        :return: GeometryFrame with new wkt column
        """
        self.frame["wkt"] = self.frame["geometry"].apply(lambda x: x.wkt)

        return self.__class__(self.frame, "geometry")

    @classmethod
    def from_file(cls, path, driver="ESRI Shapefile"):
        geometry = gpd.read_file(path, driver=driver)
        GeoFrame = cls(geometry, "geom")
        GeoFrame.crs = geometry.crs
        return GeoFrame

    def union(self, attribute):
        print(self.frame.columns)
        dissolved = self.frame.dissolve(by=attribute, aggfunc='sum')
        geoframe = self.__class__(dissolved, "geometry")
        geoframe.type = "Multi" + self.type
        return geoframe

    def _assert_geom_type(self):
        unique_geometries = [el for el in set(self.frame.type) if el is not None]
        if unique_geometries.__len__() != 1:
            raise GeometryCollectionError("Object can not be collection of geometries")

        try:
            assert str(list(set(self.frame.type))[0]) + "Frame" == self.type
        except AssertionError:
            raise GeometryTypeError("Your input geometry type is incorrect")


class PointFrame(GeometryFrame):

    def __init__(self, frame, geometry_column):
        super().__init__(frame, geometry_column)


class LineFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)


class PolygonFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)


@attr.s
class Point:

    x = attr.ib(default=0, validator=[IsNumeric()])
    y = attr.ib(default=0, validator=[IsNumeric()])

    def __str__(self):
        return f"Point({self.x} {self.y})"


@attr.s
class Line:
    start = attr.ib()
    end = attr.ib()

    def __attr_post_init__(self):
        self.dx = count_delta(self.start.x, self.start.x)
        self.dy = count_delta(self.start.y, self.end.y)


@attr.s
class Origin(Point):

    x = attr.ib(default=0)
    y = attr.ib(default=0)

    def __attr__post_init__(self):
        super().__init__(self.x, self.y)



if __name__ == "__main__":
    pointa = Point("s", 1.0)