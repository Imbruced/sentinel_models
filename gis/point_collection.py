from typing import List

import attr

from gis.extent import Extent
from gis.point import Point
from utils.decorators import lazy_property


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
