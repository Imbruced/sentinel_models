import inspect
import re
import sys
from abc import ABC
from typing import Dict

import attr
import gdal
from PIL import Image

from exceptions import OptionNotAvailableException
from gis.raster_components import ImageOptions, Options
from logs import logger


@attr.s
class ClsFinder:

    name = attr.ib(type="str")

    def __get_cls_tuples(self):
        return inspect.getmembers(sys.modules[self.name], inspect.isclass)

    @property
    def available_cls(self):
        __cls = []
        for cls in self.__get_cls_tuples():
            try:
                __cls.append(cls[1])
            except IndexError:
                pass
        return __cls


@attr.s
class Writer(ABC):
    data = attr.ib()
    writer_options = attr.ib(type=Options)

    def save(self, path: str):
        raise NotImplemented()

    def options(self, **kwargs):
        for key in kwargs:
            self.writer_options[key] = kwargs[key]

        return self.__class__(self.data, Options({**kwargs}))

    def __get_writers(self, regex) -> Dict[str, str]:
        return {cl.format_name: cl
                for cl in ClsFinder(__name__).available_cls
                if re.match(regex, cl.__name__)
                }

    def available_writers(self, regex):
        if not hasattr(self, "__writers"):
            setattr(self, "__writers", self.__get_writers(regex))
        return getattr(self, "__writers")


@attr.s
class ImageWriter(Writer):
    data = attr.ib()
    writer_options = attr.ib(default=ImageOptions())

    def save(self, path: str):
        image_format = self.__get_writer()
        writer = self.writers[image_format](
            writer_options=self.writer_options,
            data=self.data
        )
        writer.save(path)

    def __get_writer(self):
        image_format = self.writer_options["format"]
        if image_format not in self.writers:
            raise OptionNotAvailableException(f"Option {image_format} is not implemented \n available options {self.__str_writers}")
        return image_format

    @property
    def writers(self):
        return super().available_writers(r"(\w+)ImageWriter")

    @property
    def __str_writers(self):
        return ", ".join(super().available_writers(r"(\w+)ImageWriter"))


@attr.s
class GeoTiffImageWriter:
    format_name = "geotiff"
    data = attr.ib()
    writer_options = attr.ib()

    def save(self, path: str):
        drv = gdal.GetDriverByName("GTiff")
        raster = self.data
        band_number = raster.array.shape[2]
        array = raster.array
        logger.info(self.writer_options)
        dtype = self.writer_options["dtype"]
        logger.info(dtype)

        ds = drv.Create(path, array.shape[1], array.shape[0], band_number, dtype)
        transformed_ds = raster.transform(ds, raster.extent.origin, raster.pixel)
        for band in range(array.shape[2]):
            transformed_ds.GetRasterBand(band + 1).WriteArray(array[:, :, band])

@attr.s
class PngImageWriter:
    format_name = "png"
    data = attr.ib()
    writer_options = attr.ib()

    def save(self, path: str):

        im = Image.fromarray(self.data.array[:, :, 0])
        im.save(path)


