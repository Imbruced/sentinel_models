import numpy as np
from geom.data_types import Point

scriptDirectory = os.path.dirname(os.path.realpath(__file__))

class RasterShape:

    def __init__(self, extent, grid_size):
        self.__extent = extent
        self.__grid_size = grid_size
        self.__x_shape = self.__extent.dx
        self.__y_shape = self.__extent.dy
        print(type(self.__x_shape))

    @property
    def x_shape(self):
        return self.__x_shape

    @property
    def y_shape(self):
        return self.__y_shape

    @property
    def size(self):
        return self.__x_shape * self.__y_shape

class Crs:
    pass


class Extent:

    def __init__(self, left_down: Point, right_up: Point):
        self.__left_down = left_down
        self.__right_up = right_up
        self.__dx = self.__right_up.dx(self.__left_down)
        self.__dy = self.__right_up.dy(self.__left_down)

    @property
    def dx(self):
        return self.__dx

    @property
    def dy(self):
        return self.__dy

class Raster:

    def __init__(self, raster_shape: RasterShape):
        self.__raster_shape = raster_shape

    @classmethod
    def from_geometry(cls, attribute: str, RasterShape: RasterShape):
        return cls()

    def prepare_array(self):
        ones_table = np.ones((self.__raster_shape.x_shape, self.__raster_shape.y_shape))
        return ones_table

    @property
    def array(self):
        return self.prepare_array()


