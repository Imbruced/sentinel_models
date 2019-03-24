from abc import ABC
import geopandas as gpd
import os
from gis.exceptions import GeometryCollectionError
from gis.exceptions import GeometryTypeError
scriptDirectory = os.path.dirname(os.path.realpath(__file__))
from gis.descriptors import NumberType


class Origin:
    __x = NumberType()
    __y = NumberType()

    def __init__(self, x: float , y: float):
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y


class Color:
    pass


class GeometryFrame(ABC):
    _CRS = None

    def __init__(self, frame, geometry_column):
        self.__frame = frame
        self.__geometry_column = geometry_column
        self.type = self.__class__.__name__
        # if self.type != "GeometryFrame":
        #     self._assert_geom_type()

    def to_wkt(self):
        self.__frame["wkt"] = self.__frame["geometry"].apply(lambda x: x.wkt)

        return self.__class__(self.__frame, "geometry")

    @classmethod
    def from_file(cls, path, driver="ESRI Shapefile"):
        geometry = gpd.read_file(path, driver=driver)
        GeoFrame = cls(geometry, "geom")
        GeoFrame.crs = geometry.crs
        return GeoFrame

    def union(self, attribute):
        print(self.__frame.columns)
        dissolved = self.__frame.dissolve(by=attribute, aggfunc='sum')
        GeoFrame = self.__class__(dissolved, "geometry")
        GeoFrame.type = "Multi" + self.type
        return GeoFrame

    def _assert_geom_type(self):
        unique_geometries = [el for el in set(self.__frame.type) if el is not None]
        if unique_geometries.__len__() != 1:
            raise GeometryCollectionError("Object can not be collection of geometries")

        try:
            assert str(list(set(self.__frame.type))[0]) + "Frame" == self.type
        except AssertionError:
            raise GeometryTypeError("Your input geometry type is incorrect")

    @property
    def crs(self):
        return self._CRS

    @crs.setter
    def crs(self, value):
        self._CRS = value

    @property
    def frame(self):
        return self.__frame


class PointFrame(GeometryFrame):

    def __init__(self, frame, geometry_column):
        super().__init__(frame, geometry_column)


class LineFrame(GeometryFrame):
    def __init__(self, frame, geometry_column):
        super().__init__(frame, geometry_column)


class PolygonFrame(GeometryFrame):
    def __init__(self, frame, geometry_column):
        super().__init__(frame, geometry_column)


class Coordinate:

    def __init__(self, value, crs):
        self.value = value
        self.crs = crs
        self.__name = "Coordinate"

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value


class X(Coordinate):

    def __init__(self, value, crs):
        super().__init__(value, crs)
        self.name = "X"


class Y(Coordinate):

    def __init__(self, value, crs):
        super().__init__(value, crs)
        self.name = "Y"


class Point:

    def __init__(self, X, Y):
        self.__x = X
        self.__y = Y

    def __str__(self):
        return f"Point({self.__x.value} {self.__y.value})"

    def count_delta(self, val1, val2):
        return val2 - val1

    def dx(self, cls):
        if not hasattr(self, "__deltax"):
            setattr(self, "__deltax", self.count_delta(cls.x, self.x))
        return getattr(self, "__deltax")

    def dy(self, cls):
        if not hasattr(self, "__deltay"):
            setattr(self, "__deltay", self.count_delta(cls.y, self.y))
        return getattr(self, "__deltay")

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y


