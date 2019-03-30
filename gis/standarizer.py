import attr
import json
import numpy as np
from gis.log_lib import logger
from gis.raster_components import ReferencedArray
import os
from gis.raster_components import Raster


class StanarizeParams:

    _band_number = attr.ib()
    _coefficients = {}

    def __init__(self, band_number):
        self._band_number = band_number

    def add(self, coeff, band_name):
        if len(self._coefficients.keys()) >= self._band_number:
            raise OverflowError("Value is to high")
        else:
            try:
                self._coefficients.get(band_name)
            except KeyError:
                self._coefficients[band_name] = coeff

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        raise ValueError("Value can not be set, please use add method")


class ImageStand:
    """
    Class is responsible for scaling image data, it requires gis.raster.Raster object
    Band names can be passed, by default is ["band_index" ... "band_index+n"]
    use standarize_image method to standarize data, it will return Raster with proper refactored image values

    """

    def __init__(self, raster, length, names=None):
        self.__stan_params = StanarizeParams(length)
        self.__names = [f"band_{band}" for band in range(length)] if names is None else names
        logger.info(self.__names)
        self.__raster = raster

    def save_params(self, path):
        if os.path.exists(path):
            raise FileExistsError("File with this name exists in directory")
        with open(path, "w") as file:
            params_json = json.load(self.__stan_params.coefficients)
            file.writelines(params_json)

    def standarize_image(self, rescale_function=None):
        """
        Rescale data by maximum
        :return:
        """
        empty_array = np.empty(shape=[*self.__raster.ref.shape, self.__raster.ref.band_number])

        for index, (band, name) in enumerate(zip(self.__raster.array, self.__names)):
            empty_array[:, :, index] = self.__stand_one_dim(band, name, rescale_function)
            band = None

        ref = ReferencedArray(empty_array, self.__raster.ref.extent, self.__raster.pixel, shape=self.__raster.ref.shape, band_number=self.__raster.ref.band_number)

        return Raster(self.__raster.pixel, ref)

    def __stand_one_dim(self, array: np.ndarray, band_name: str, rescale_function=None):
        """
        Standarizing 2 dim array, fnct for rescaling can be passed: require function which takes array and returns
        array
        :param array:
        :return:
        """
        if rescale_function is None:
            stand_value = self.find_max(array)
        else:
            stand_value = rescale_function(array)

        self.__stan_params.add(stand_value, band_name)
        return array/stand_value

    @staticmethod
    def find_max(array: np.ndarray):
        if type(array) != np.ndarray:
            logger.error("Inappropriate type")
            raise TypeError(f"Argument should be numpy ndarray, type {type(array)} found")

        return array.max()

    @property
    def stan_params(self):
        return self.__stan_params

    @stan_params.setter
    def stan_params(self, value):
        raise AttributeError("This param can' t be set")
