import attr
from shapely.geometry import Point as ShapelyPoint

from validators.validators import IsNumeric


@attr.s
class Point(ShapelyPoint):
    TYPE = "Point"
    _x = attr.ib(default=0, validator=[IsNumeric()])
    _y = attr.ib(default=0, validator=[IsNumeric()])

    def __attrs_post_init__(self):
        super().__init__(self._x, self._y)

    def __str__(self):
        return f"Point({self.x} {self.y})"

    def translate(self, dx, dy):
        """
        Translates points coordinates by x and y value
        :param x: number of units to move along x axis
        :param y: number of units to move along y axis
        :return: new Point with translated coordinates
        """
        return Point(self.x + dx, self.y + dy)
