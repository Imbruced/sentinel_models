import attr

from validators.validators import ispositive
import os
from meta.meta import ConfigMeta
import typing


@attr.s
class Pixel(metaclass=ConfigMeta):
    x = attr.ib(default=1.0, validator=[attr.validators.instance_of(float), ispositive])
    y = attr.ib(default=1.0, validator=[attr.validators.instance_of(float), ispositive])
    unit = attr.ib(default='m', validator=[attr.validators.instance_of(str)])

    @classmethod
    def from_text(cls, text):
        x, y, unit = text.split(" ")
        return cls(int(x), int(y), unit)


@attr.s
class ReferencedArray:
    array = attr.ib()
    extent = attr.ib()
    crs = attr.ib()
    shape = attr.ib()
    is_generator = attr.ib(default=False)
    band_number = attr.ib(default=1)


@attr.s
class Path:

    path_string: str = attr.ib()

    def __attrs_post_init__(self):
        pass

    def is_file(self):
        return os.path.isfile(self.path_string)

    def __exists(self, path):
        pass

    def create(cls, path):
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0], exist_ok=True)


@attr.s
class ArrayShape:
    shape = attr.ib()

    def __attrs_post_init__(self):
        self.x_shape = self.shape[0]
        self.y_shape = self.shape[1]

    def __ne__(self, other):
        return self.x_shape != other.x_shape or self.y_shape != other.y_shape


def create_chunk(iterable: typing.Iterable, chunk_size: int):
    for chunk in range(0, len(iterable), chunk_size):
        yield iterable[chunk: chunk + chunk_size]


def create_two_dim_chunk(iterable: typing.Iterable, chunk_size: typing.List[int]):
    for el in create_chunk(iterable, chunk_size[0]):
        yield from (tel.transpose(1, 0, 2) for tel in create_chunk(el.transpose([1, 0, 2]), chunk_size[1])
                    if tel.shape[1] == chunk_size[0] and tel.shape[0] == chunk_size[1])


