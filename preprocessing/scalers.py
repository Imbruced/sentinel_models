import attr
import numpy as np

@attr.s
class MaxScaler:
    copy = attr.ib(default=True)
    value = attr.ib(default=None, validator=[])

    def fit(self, array: np.ndarray):
        if type(array) != np.ndarray:
            raise TypeError("This method accepts only numpy array")
        maximum_value = array.max()
        return MaxScaler(copy=True, value=maximum_value)

    def fit_transform(self, array: np.ndarray):
        self.value = array.max()
        return array/self.value

    def transform(self, array: np.ndarray):
        if self.value is not None:
            return array/self.value
        else:
            raise ValueError("Can not divide by None")


@attr.s
class RangeScaler:
    copy = attr.ib(default=True)
    value = attr.ib(default=None, validator=[])

    def fit(self, array):
        maximum_value = array.max() - array.min()
        return self.__class__(copy=True, value=maximum_value)

    def fit_transform(self, array: np.ndarray):
        array_min = array.min()
        array_max = array.max()
        self.value = array_max - array_min
        return (array-array_min)/self.value

    def transform(self, array: np.ndarray):
        array_min = array.min()
        if self.value is not None:
            return (array-array_min)/self.value
        else:
            raise ValueError("Can not divide by None")


@attr.s
class StanarizeParams:

    band_number = attr.ib()
    coefficients = attr.ib(default={})

    def add(self, coeff, band_name):
        if band_name not in self.coefficients.keys():
            self.coefficients[band_name] = coeff
        else:
            self.coefficients[band_name] = coeff