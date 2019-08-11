import json

import attr
import numpy as np

from gis.raster_components import ReferencedArray
from preprocessing.scalers import StanarizeParams


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
        self.names = [f"band_{band}" for band in range(self.raster.shape[-1])] \
            if self.names is None else self.names

    def save_params(self, path):
        if os.path.exists(path):
            raise FileExistsError("File with this name exists in directory")
        with open(path, "w") as file:
            params_json = json.load(self.stan_params.coefficients)
            file.writelines(params_json)

    def standarize_image(self, scaler=None):
        from gis.raster import Raster
        """
        Rescale data by maximum
        :return:
        """
        empty_array = np.empty(shape=[*self.raster.shape])

        for index, name in enumerate(self.names):
            empty_array[:, :, index] = self.__stand_one_dim(self.raster[:, :, index], name, scaler)
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

