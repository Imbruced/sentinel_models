import attr
from gis.exceptions import LessThanZeroException
from gis.log_lib import logger


def ispositive(instance, attribute, value):
    if value <= 0:
        raise LessThanZeroException("Value has to be bigger than 0")


def is_in():
    pass