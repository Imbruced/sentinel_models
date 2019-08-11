from typing import Dict, List, Type

import attr

from abstract.io_handler import ReadAbstractFactory
from config.options import Options
from config.options_read import DefaultOptionRead
from exceptions import OptionNotAvailableException
from exceptions.exceptions import FormatNotAvailable
from readers.reader import Reader
from writers.writer import Writer


@attr.s
class ImageReaderFactory(ReadAbstractFactory):

    io_options = attr.ib(default=None)

    def load(self, path) -> 'Raster':
        if self.io_options is None:
            raise AttributeError("Please use format first")
        image_format = self.__get_reader()
        raster: 'Raster' = self.readers[image_format](
            io_options=self.io_options,
            path=path
        ).load()

        return raster

    def format(self, format: str):
        try:
            default_options = getattr(DefaultOptionRead, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")
        return self.__class__(
            io_options=default_options
        )

    @property
    def readers(self) -> Dict[str, Type['Reader']]:
        return {cl.format_name: cl for cl in self.available_cls()}

    @property
    def __str_readers(self):
        return ", ".join([self.readers[key].format_name for key in self.readers])

    def get_cls(self, name: str) -> Type['Reader']:
        return self.readers[name]

    def available_cls(self) -> List[Type['Reader']]:
        from readers import readers
        return readers

    def __get_reader(self):
        """TODO to simplify or move to upper class"""
        image_format = self.io_options["format"]
        if image_format not in self.readers:
            raise OptionNotAvailableException(f"Option {image_format} is not implemented \n available options {self.__str_readers}")
        return image_format

    def options(self, **kwargs) -> 'ImageReaderFactory':
        io_options = super().options(**kwargs).io_options
        return ImageReaderFactory(io_options)
