import gdal
import attr
import numpy as np

from validators.validators import IsImageFile
from gis.raster_components import Pixel
from gis.geometry import Extent


@attr.s
class ImageFile:

    """
    Class prepared for image path validation
    """

    path = attr.ib(validator=[IsImageFile()])

    def __str__(self):
        return self.path


@attr.s
class GdalImage:

    ds = attr.ib(type=gdal.Dataset)
    path = attr.ib()
    crs = attr.ib(default="local")

    def __attrs_post_init__(self):
        self.__transform_params = self.ds.GetGeoTransform()
        self.left_x = self.__transform_params[0]
        self.pixel_size_x = self.__transform_params[1]
        self.top_y = self.__transform_params[3]
        self.pixel_size_y = -self.__transform_params[5]
        self.x_size = self.ds.RasterXSize
        self.y_size = self.ds.RasterYSize
        self.pixel = Pixel(self.pixel_size_x, self.pixel_size_y)
        self.extent = Extent.from_coordinates([
            self.left_x, self.top_y - (self.y_size * self.pixel_size_y),
            self.left_x + self.pixel_size_x * self.x_size, self.top_y
        ], self.crs)
        self.band_number = self.ds.RasterCount

    @classmethod
    def load_from_file(cls, path: str, crs: str):
        """
        class method which based on path returns GdalImage object,
        It validates path location and its format
        :return: GdalImage instance
        """
        file = ImageFile(path)
        ds: gdal.Dataset = gdal.Open(file.path)

        return cls(ds, path, crs)

    def __read_as_array(self, ds: gdal.Dataset) -> np.ndarray:
        if not hasattr(self, "__array"):
            setattr(self, "__array", ds.ReadAsArray())
        return getattr(self, "__array")

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    @property
    def array(self) -> np.ndarray:
        """
        Property which returns numpy.ndarray representation of file
        :return: np.ndarray
        """
        return self.__read_as_array(self.ds)

    def save(self, path):
        pass



