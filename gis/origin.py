import attr

from gis.point import Point


@attr.s
class Origin(Point):
    _x = attr.ib(default=0)
    _y = attr.ib(default=0)

    def __attr__post_init__(self):
        super().__init__(self._x, self._y)
