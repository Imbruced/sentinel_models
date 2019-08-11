from typing import List

import attr

from gis.crs import Crs
from gis.geometry_operations import count_delta
from gis.origin import Origin
from gis.point import Point


@attr.s
class Extent:
    left_down = attr.ib(default=Point(0, 0))
    right_up = attr.ib(default=Point(1, 1))
    crs = attr.ib(default=Crs("epsg:4326"))
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
        scaled_point = Point(int(self.dx / x + origin.x), int(self.dy / y + origin.y))
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

    def expand(self, dx, dy):
        ld = self.left_down.translate(-dx, -dy)
        ru = self.right_up.translate(dx, dy)

        return Extent(ld, ru, crs=self.crs)

    def expand_percentage(self, percent_x, percent_y):
        return self.expand(int(self.dx * percent_x), int(self.dy * percent_y))

    def expand_percentage_equally(self, percent):
        return self.expand_percentage(percent, percent)

    def expand_equally(self, value):
        return self.expand(value, value)

    def to_wkt(self):
        coordinates = [
            self.left_down,
            self.left_down.translate(0, self.dy),
            self.right_up,
            self.left_down.translate(self.dx, 0),
            self.left_down
        ]

        coordinates_text = ", ".join([f"{el.x} {el.y}" for el in coordinates])
        return f"POLYGON(({coordinates_text}))"

    def divide_dy(self, tile_size):
        tiles_number_dy = int(float(self.dy) // float(tile_size))
        extents = []
        for tile in range(0, tiles_number_dy):
            extents.append(
                Extent(
                    self.right_up.translate(-self.dx, -(tile + 1) * tile_size),
                    self.right_up.translate(0, (-tile) * tile_size),
                    crs=self.crs)
            )
        if int(float(self.dy) // float(tile_size)) != float(self.dy) / float(tile_size):
            extents.append(Extent(
                self.right_up.translate(-self.dx, -self.dy),
                self.right_up.translate(0, -tiles_number_dy * tile_size),
                self.crs
            ))
        return extents

    def divide_dx(self, tile_size):
        tiles_number_dx = int(float(self.dx) // float(tile_size))
        extents = []

        for tile in range(tiles_number_dx):
            extents.append(
                Extent(
                    self.left_down.translate(tile * tile_size, 0),
                    self.left_down.translate((tile + 1) * tile_size, self.dy),
                    crs=self.crs)
            )
        if int(float(self.dx) // float(tile_size)) == float(self.dx) / float(tile_size):
            extents.append(Extent(
                self.left_down.translate(tiles_number_dx * tile_size, 0),
                self.left_down.translate(self.dx, self.dy),
                self.crs
            ))

        return extents

    def divide(self, dx, dy):

        if all([dx, dy]):
            dy_divided = self.divide_dy(dy)
            extents = []
            for dy_tile in dy_divided:
                dx_divided = dy_tile.divide_dx(dx)
                for dx_tile in dx_divided:
                    extents.append(dx_tile)
            return extents
        else:
            raise AttributeError("You have to pass all the arguments")
        pass
