import attr
from gis.exceptions import LessThanZeroException
from gis.exceptions import CrsException
from gis.crs import CRS
import os
from gis.exceptions import ExtensionException

image_file_extensions = [
    "png",
    "tif",
    "tiff",
    "geotiff",
    "jpg",
    "jpeg"
]


def ispositive(instance, attribute, value):
    if value <= 0:
        raise LessThanZeroException("Value has to be bigger than 0")


@attr.s
class IsNumeric:

    def __call__(self, instance, attribute, value):
        current_type = type(value)
        if current_type not in [int, float]:
            raise TypeError(f"Value should be float or integer but is {current_type}")

@attr.s
class IsPositiveNumeric(IsNumeric):
    pass


class IsCrs:

    def __call__(self, instance, attribute, value):
        if value not in CRS:
            raise CrsException("This is not valid coordinate reference system")


@attr.s
class IsFile:

    def __call__(self, instance, attribute, value):
        if not os.path.isfile(value):
            raise FileExistsError("This is not file")


@attr.s
class IsImageFile(IsFile):

    def __call__(self, instance, attribute, value):
        super().__call__(instance, attribute, value)
        path, file_name = os.path.split(value)

        try:
            extension = file_name.split(".")[1]
        except IndexError:
            raise ExtensionException("File has not clearly specified extension")

        if extension.lower() not in image_file_extensions:
            raise ExtensionException("File has inappropriate extension")


def validate_shapes(x, y):
    pass


class UnetImageShape:

    def __call__(self, instance=None, attribute=None, value=None):
        if not isinstance(value, tuple) and not isinstance(value, list):
            raise TypeError("Shape should be list or tuple")
        if len(value) != 3:
            raise AttributeError("Length should be 3")

        for val in value:
            if not isinstance(val, int):
                raise TypeError("Value should be integer")

        dimension = value[-1]

        if dimension <= 0:
            raise AttributeError("Dimension should be >= 0")

        shapes = value[:-1]

        if not all([self.__validate_dimension(val) for val in shapes]):
            raise ValueError("Dimension should be multiplicity of 2 and >= 16")


    def __validate_dimension(self, value):
        return not value % 2 and value > 16


