from abc import ABC

import attr

from exceptions.exceptions import FormatNotAvailable
from gis.io_abstract import DefaultOptionRead
from gis.raster_components import Options
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

    def load(self):
        pass

    def readers(self):
        pass

    @property
    def __str_readers(self):
        return ", ".join(self.available_cls(r"(\w+)ImageReader", __name__))


@attr.s
class GeoTiffImageReader:
    pass


@attr.s
class ShapeImageReader:
    pass


@attr.s
class WktImageReader:
    pass


@attr.s
class PngImageReader:
    pass