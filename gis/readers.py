from abc import ABC

import attr
import gdal

from exceptions.exceptions import FormatNotAvailable, OptionNotAvailableException
from gis.image_loader import ImageFile, GdalImage
from gis.io_abstract import DefaultOptionRead
from gis.raster import Raster
from gis.raster_components import Options, ReferencedArray
from gis.writers import IoHandler


@attr.s
class Reader(IoHandler):
    io_options = attr.ib(type=Options)

    def load(self, path: str):
        return NotImplemented()

    def options(self, **kwargs):
        options = super().options(**kwargs)
        return self.__class__(options)

    def format(self, format):
        try:
            default_options = getattr(DefaultOptionRead, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")
        return self.__class__(
            io_options=default_options
        )


@attr.s
class ImageReader(Reader):

    io_options = attr.ib(type=Options, default=getattr(DefaultOptionRead, "wkt")())

    def load(self, path) -> Raster:
        image_format = self.__get_reader()
        reader = self.readers[image_format](
            io_options=self.io_options,
            path=path
        ).load()

        return reader

    @property
    def readers(self):
        return self.available_cls(r"(\w+)ImageReader", __name__)

    def __str_readers(self):
        return ", ".join(self.available_cls(r"(\w+)ImageReader", __name__))

    def __get_reader(self):
        """TODO to simplify or move to upper class"""
        image_format = self.io_options["format"]
        if image_format not in self.readers:
            raise OptionNotAvailableException(f"Option {image_format} is not implemented \n available options {self.__str_readers}")
        return image_format


@attr.s
class GeoTiffImageReader:

    path = attr.ib()
    io_options = attr.ib()
    format_name = "geotiff"

    def load(self):

        gdal_image = GdalImage.load_from_file(
            self.path,
            self.io_options["crs"])

        ref = ReferencedArray(array=gdal_image.array.transpose([1, 2, 0]),
                              crs=self.io_options["crs"], extent=gdal_image.extent,
                              band_number=gdal_image.band_number,
                              shape=[gdal_image.pixel_size_y, gdal_image.pixel_size_x])
        return Raster(pixel=gdal_image.pixel, ref=ref)


@attr.s
class ShapeImageReader:
    format_name = "shp"
    pass


@attr.s
class WktImageReader:
    format_name = "wkt"


@attr.s
class PngImageReader:
    format_name = "png"

