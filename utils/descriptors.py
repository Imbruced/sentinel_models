from exceptions.exceptions import LessThanZeroException
from abc import ABC


class InstanceDecorator(ABC):

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name


class PositiveValue(InstanceDecorator):
    """
    This is descriptor which allows to correctly handle shape of array
    """

    def __set__(self, instance, value):
        if value < 0:
            raise LessThanZeroException("Shape value has to be more than 0")
        super().__set__(instance, value)


class NumberType(InstanceDecorator):

    def __set__(self, instance, value):
        if type(value) not in [float, int]:
            raise ValueError("Value must be numeric")
        super().__set__(instance, value)


class PositiveInteger(InstanceDecorator):

    def __set__(self, instance, value):
        try:
            value = float(value)
        except TypeError:
            raise ValueError("Value has to be numeric")

        if value % 1 == 0 and value > 0:
            super().__set__(instance, value)
        else:
            raise ValueError("Value must be integer bigger than 0")








