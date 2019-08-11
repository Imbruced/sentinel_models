import re

import attr
from shapely import wkt as wkt_loader

from gis.extent import Extent
from gis.geometry import cast_float
from gis.point import Point
from gis.point_collection import PointCollection
from utils.decorators import lazy_property
from validators.validators import WktValidator


@attr.s
class Wkt:
    wkt_string = attr.ib(type=str, validator=WktValidator())
    crs = attr.ib(default="local")

    def __attrs_post_init__(self):
        self.__geom = wkt_loader.loads(self.wkt_string)
        self.__type = self.__geom.type

    #
    # def to_geometry(self):
    #     return shapely.wkt.loads(self.wkt_string)

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