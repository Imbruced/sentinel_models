import attr

from config.options import Options
from abstract.io_handler import IoHandler


@attr.s
class Reader(IoHandler):
    io_options = attr.ib(type=Options)

    def load(self):
        return NotImplemented()

    @property
    def format_name(self):
        raise NotImplementedError

    @format_name.setter
    def format_name(self, value: str):
        raise NotImplementedError
