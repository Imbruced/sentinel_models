from abc import ABC
from gis.raster import Raster
import geopandas as gpd
import os

scriptDirectory = os.path.dirname(os.path.realpath(__file__))


class GeometryFrame(ABC):

    def __init__(self, path: str, driver="shapefile"):
        self.__path = path
        self.__driver = driver

    def load_file(self):
        geometry = gpd.read_file(self.__path,
                                 driver=self.__driver)
        self.crs = geometry.crs
        return geometry

    def to_raster(self, attribute, grid_size):
        return Raster().from_geometry(attribute, grid_size)

    def union(self):
        pass

    @property
    def geometry(self):
        if not hasattr(self, "__geometry"):
            setattr(self, "__geometry", self.load_file())
            return getattr(self, "__geometry")
        else:
            return getattr(self, "__geometry")

    @property
    def extent(self):
        pass


class PointFrame(GeometryFrame):
    pass


class LineFrame(GeometryFrame):
    pass


class PolygonFrame(GeometryFrame):
    pass


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
