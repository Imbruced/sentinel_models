import abc
from typing import Dict

import attr

from config.options import Options
from config.options_write import DefaultOptionWrite
from exceptions.exceptions import FormatNotAvailable
from abstract.io_handler import IoHandler


@attr.s
class Writer(IoHandler):
    data = attr.ib()
    io_options = attr.ib(type=Options())

    def save(self, path: str):
        raise NotImplemented()

    def options(self, **kwargs):
        current_options = super().options(**kwargs)
        return self.__class__(data=self.data, io_options=current_options)

    def format(self, format: str):
        try:
            default_options = getattr(DefaultOptionWrite, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")

        return self.__class__(
            data=self.data,
            io_options=default_options
        )

    @property
    def format_name(self):
        raise NotImplementedError()

    @format_name.setter
    def format_name(self, value: str):
        raise NotImplementedError
