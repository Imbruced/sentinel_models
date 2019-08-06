from typing import NoReturn, Any, Tuple

import attr
import numpy as np

from gis import Pixel
from gis.crs import Crs
from gis.gdal_image import GdalImage
from gis.geometry import lazy_property, Extent
from gis.raster_components import ReferencedArray
from readers.image import ImageReader
from utils.decorators import classproperty
from writers.image import ImageWriter


@attr.s
class RasterCreator:

    @classmethod
    def empty_raster(cls, extent: Extent, pixel: Pixel) -> Tuple[Pixel, ReferencedArray]:
        pass

    @staticmethod
    def to_raster(gdal_raster: GdalImage, pixel: Pixel) -> Tuple[Pixel, ReferencedArray]:
        pass


class Raster(np.ndarray):

    def __new__(cls, pixel: Pixel, ref: ReferencedArray) -> Any:
        pass

    def __array_finalize__(self, obj) -> Any:
        pass

    def __array_wrap__(self, out_arr, context=None) -> Any:
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> 'Raster':
        pass

    @classproperty
    def read(self) -> 'ImageReader':
        pass

    @property
    def write(self) -> 'ImageWriter':
        pass

    @classmethod
    def from_array(cls, array, pixel: Pixel, extent: Extent) -> 'Raster':
        pass

    @classmethod
    def empty(cls, extent: Extent, pixel: Pixel) -> 'Raster':
        pass

    @property
    def array(self) -> np.ndarray:
        return self.ref.array

    def show(self, true_color=False) -> NoReturn:
        pass

    @lazy_property
    def extent(self) -> Extent:
        pass

    @lazy_property
    def crs(self) -> Crs:
        pass