from abc import ABC
from copy import deepcopy
from typing import NoReturn

import attr


@attr.s
class IoAbstractFactory(ABC):
    io_options = attr.ib(default=None)

    def format(self, format: str) -> 'IoAbstractFactory':
        raise NotImplementedError

    def options(self, **kwargs) -> 'IoAbstractFactory':
        current_options = deepcopy(self.io_options)
        for key in kwargs:
            current_options[key] = kwargs[key]
        return self.__class__(current_options)


@attr.s
class ReadAbstractFactory(IoAbstractFactory):
    io_options = attr.ib(default=None)

    def load(self, path: str) -> 'Raster':
        raise NotImplementedError

    def format(self, format: str) -> 'ReadAbstractFactory':
        raise NotImplementedError

    def options(self, **kwargs) -> 'ReadAbstractFactory':
        io_options = super().options(**kwargs).io_options
        return ReadAbstractFactory(io_options)


@attr.s
class WriteAbstractFactory(IoAbstractFactory):
    io_options = attr.ib(default=None)
    data = attr.ib(default=None)

    def save(self, path: str) -> NoReturn:
        raise NotImplementedError

    def format(self, path: str) -> 'WriteAbstractFactory':
        raise NotImplementedError

    def options(self, **kwargs) -> 'WriteAbstractFactory':
        io_options = super().options(**kwargs).io_options
        return WriteAbstractFactory(data=self.data, io_options=io_options)


@attr.s
class IoHandler(ABC):
    io_options = attr.ib()

    @property
    def format_name(self):
        raise NotImplementedError

    @format_name.setter
    def format_name(self, value: str):
        raise NotImplementedError
