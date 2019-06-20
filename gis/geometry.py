import re
from abc import ABC
import os
from typing import List

import attr
import geopandas as gpd


from gis.geometry_operations import count_delta
from validators.validators import IsNumeric, WktValidator
from exceptions.exceptions import GeometryCollectionError
from exceptions.exceptions import GeometryTypeError


scriptDirectory = os.path.dirname(os.path.realpath(__file__))


def lazy_property(fn):
    attr_name = '__' + fn.__name__

    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return wrapper


@attr.s
class GeometryFrame(ABC):

    frame = attr.ib()
    geometry_column = attr.ib()
    crs = attr.ib(default="local", validator=[])

    def __attr__post_init__(self):
        self.type = self.__class__.__name__

    def to_wkt(self):
        """
        Based on geometry column this method returns wkt representation of it, it does not modify
        passed instance, it create new one

        :return: GeometryFrame with new wkt column
        """

        frame_copy = self.frame.copy()
        frame_copy["wkt"] = frame_copy["geometry"].apply(lambda x: x.wkt)

        return self.__class__(frame_copy, "geometry")

    @classmethod
    def from_file(cls, path, driver="ESRI Shapefile"):
        """
        TODO properly handle crs from file
        :param path:
        :param driver:
        :return:
        """

        geometry = gpd.read_file(path, driver=driver)
        GeoFrame = cls(geometry, "geom")
        GeoFrame.crs = geometry.crs["init"]
        return GeoFrame

    def union(self, attribute):
        dissolved = self.frame.dissolve(by=attribute, aggfunc='sum')
        geoframe = self.__class__(dissolved, "geometry")
        geoframe.type = "Multi" + self.type
        geoframe.crs = self.crs
        return geoframe

    # def transform(self, to_epsg, from_epsg=None):
    #     if self.crs is None or self.crs == "":
    #         self.crs =

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
        super().__attr__post_init__()


class LineFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


class PolygonFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


@attr.s
class Point:

    x = attr.ib(default=0, validator=[IsNumeric()])
    y = attr.ib(default=0, validator=[IsNumeric()])

    def __str__(self):
        return f"Point({self.x} {self.y})"

    def translate(self, x, y):
        """
        Translates points coordinates by x and y value
        :param x: number of units to move along x axis
        :param y: number of units to move along y axis
        :return: new Point with translated coordinates
        """
        return Point(self.x + x, self.y + y)


@attr.s
class PointCollection:
    points = attr.ib(factory=list, type=List[Point])
    crs = attr.ib(default="local")

    def __max_y(self):
        return max([el.y for el in self.points])

    def __max_x(self):
        return max([el.x for el in self.points])

    def __min_x(self):
        return min([el.x for el in self.points])

    def __min_y(self):
        return min([el.y for el in self.points])

    @lazy_property
    def extent(self):
        return Extent(
            Point(self.__min_x(), self.__min_y()),
            Point(self.__max_x(), self.__max_y())
        )


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


@attr.s
class Extent:

    left_down = attr.ib(default=Point(0, 0))
    right_up = attr.ib(default=Point(1, 1))
    crs = attr.ib(default="local")
    origin = attr.ib(init=False)
    dx = attr.ib(init=False)
    dy = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.origin = Origin(self.left_down.x, self.left_down.y)
        self.dx = count_delta(self.left_down.x, self.right_up.x)
        self.dy = count_delta(self.left_down.y, self.right_up.y)

    def transform(self, to_crs):

        pass

    def scale(self, x, y, origin=Point(0, 0)):
        """
        This function takes x and y as the scaling values and divide extent dx and dy by them
        If origin Point is not passed by default it is Point(0, 0)
        :param x: Scaling value x
        :param y: Scaling value y
        :param origin: is the left down corner from which scaled extent will have origin
        :return: returns New instance of extent
        """
        scaled_point = Point(int(self.dx/x + origin.x), int(self.dy/y + origin.y))
        shrinked = Extent(origin, scaled_point)

        return shrinked

    def translate(self, x, y):
        """
        Translates extent coordinates
        :param x:
        :param y:
        :return:
        """

        return Extent(self.left_down, self.right_up.translate(x, y))

    @classmethod
    def from_coordinates(cls, coordinates: List[float], crs="local"):
        point_a = Point(*coordinates[:2])
        point_b = Point(*coordinates[2:])

        return cls(point_a, point_b, crs)


def cast_float(string: str):
    try:
        casted = float(string)
    except TypeError:
        raise TypeError(f"Can not cast to float value {string}")
    return casted


@attr.s
class Wkt:

    wkt_string = attr.ib(type=str, validator=WktValidator())
    crs = attr.ib(default="local")

    def __split(self):
        return re.findall(r"(-*\d+\.*\d*) (-*\d+\.*\d*)", self.wkt_string)

    def get_type(self):
        pass

    @lazy_property
    def extent(self) -> Extent:
        return PointCollection(self.coordinates, crs=self.crs).extent

    @lazy_property
    def coordinates(self):
        try:
            coordinates = [Point(cast_float(el[0]), cast_float(el[1])) for el in self.__split()]
        except TypeError:
            raise TypeError("Wrong wkt format")
        except IndexError:
            raise IndexError("Wrong wkt format")
        return coordinates