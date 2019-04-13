import attr
from gis.exceptions import LessThanZeroException
from gis.exceptions import CrsException
from gis.crs import CRS
import os
from gis.exceptions import ExtensionException
from gis.log_lib import logger


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

