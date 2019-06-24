import attr
import json
import numpy as np
from logs import logger
from gis.raster_components import ReferencedArray
import os
from gis.image_data import Raster


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
class StanarizeParams:

    band_number = attr.ib()
    coefficients = attr.ib(default={})

    def add(self, coeff, band_name):
        if band_name not in self.coefficients.keys():
            self.coefficients[band_name] = coeff
        else:
            logger.info("Overwriting value")
            self.coefficients[band_name] = coeff


@attr.s
class ImageStand:
    """
    Class is responsible for scaling image data, it requires gis.raster.Raster object
    Band names can be passed, by default is ["band_index" ... "band_index+n"]
    use standarize_image method to standarize data, it will return Raster with proper refactored image values

    """
    stan_params = attr.ib(init=False)
    raster = attr.ib()
    names = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.stan_params = StanarizeParams(self.raster.ref.band_number)
        self.names = [f"band_{band}" for band in range(self.raster.ref.band_number)] \
            if self.names is None else self.names

    def save_params(self, path):
        if os.path.exists(path):
            raise FileExistsError("File with this name exists in directory")
        with open(path, "w") as file:
            params_json = json.load(self.stan_params.coefficients)
            file.writelines(params_json)

    def standarize_image(self, scaler=None):
        """
        Rescale data by maximum
        :return:
        """
        logger.info(self.raster.ref.extent.dy)
        empty_array = np.empty(shape=[*self.raster.array.shape[:2],
                                      int(self.raster.ref.band_number)])
        logger.info(empty_array.shape)

        for index, name in enumerate(self.names):
            empty_array[:, :, index] = self.__stand_one_dim(self.raster.array[:, :, index], name, scaler)
            band = None

        ref = ReferencedArray(empty_array,
                              self.raster.ref.extent,
                              self.raster.pixel,
                              shape=self.raster.ref.shape,
                              band_number=self.raster.ref.band_number)

        return Raster(self.raster.pixel, ref)

    def __stand_one_dim(self, array: np.ndarray, band_name: str, scaler):
        """

        :param array:
        :param band_name:
        :param scaler:
        :return:
        """

        fitted = scaler.fit(array)

        self.stan_params.coefficients[band_name] = fitted

        self.stan_params.add(fitted, band_name)
        return fitted.transform(array)
