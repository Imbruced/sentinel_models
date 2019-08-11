from typing import Dict, List

import attr

from abstract.io_handler import WriteAbstractFactory
from config.options_write import DefaultOptionWrite
from exceptions import OptionNotAvailableException
from exceptions.exceptions import FormatNotAvailable
from writers.writer import Writer


@attr.s
class ImageWriterFactory(WriteAbstractFactory):
    data = attr.ib()
    io_options = attr.ib(default=None)

    def save(self, path: str):
        if self.io_options is None:
            raise AttributeError("Please use options first")
        image_format = self.__get_writer()
        writer = self.writers[image_format](
            io_options=self.io_options,
            data=self.data
        )
        writer.save(path)

    def format(self, format: str):
        try:
            default_options = getattr(DefaultOptionWrite, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")
        return self.__class__(
            io_options=default_options,
            data=self.data
        )

    def __get_writer(self):
        image_format = self.io_options["format"]
        if image_format not in self.writers:
            raise OptionNotAvailableException(
                f"Option {image_format} is not implemented \n available options {self.__str_writers}")
        return image_format

    @property
    def writers(self) -> Dict[str, 'Writer']:
        return {cl.format_name: cl for cl in self.available_cls()}

    @property
    def __str_writers(self):
        return ", ".join([self.writers[key].format_name for key in self.writers])

    def get_cls(self, name: str) -> 'Writer':
        return self.writers[name]

    def available_cls(self) -> List['Writer']:
        from writers import writers
        return writers

    def options(self, **kwargs) -> 'ImageWriterFactory':
        io_options = super().options(**kwargs).io_options
        return ImageWriterFactory(data=self.data, io_options=io_options)


