import attr
from gis.exceptions import LessThanZeroException
from gis.log_lib import logger
from gis.exceptions import CrsException
from gis.crs import CRS

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
