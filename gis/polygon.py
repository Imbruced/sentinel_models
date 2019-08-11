from typing import List

import attr
from shapely.geometry import Polygon as ShapelyPolygon

from gis.crs import Crs
from shapely import wkt as wkt_loader


@attr.s
class Polygon(ShapelyPolygon):
    NAME = "Polygon"
    coordinates = attr.ib(type=List[List[float]])
    crs = attr.ib(default=Crs("epsg:4326"))

    def __attrs_post_init__(self):
        super().__init__(self.coordinates)

    @classmethod
    def from_wkt(cls, wkt, crs):
        polygon = wkt_loader.loads(wkt)
        try:
            assert polygon.type == cls.NAME
        except AssertionError:
            raise TypeError(f"wkt is type {polygon.type}")

        return cls(polygon.exterior.coords, crs)