import attr
from gis.log_lib import logger
from gis.validators import ispositive
import numpy as np
import gdal
import os
from typing import List
from gis.meta import ConfigMeta


@attr.s
class Pixel(metaclass=ConfigMeta):
    x = attr.ib(default=1, validator=[attr.validators.instance_of(float), ispositive])
    y = attr.ib(default=0, validator=[attr.validators.instance_of(float), ispositive])
    unit = attr.ib(default='m', validator=[attr.validators.instance_of(str)])

    @classmethod
    def from_text(cls, text):
        x, y, unit = text.split(" ")
        return cls(int(x), int(y), unit)


@attr.s
class Crs(metaclass=ConfigMeta):
    epsg = attr.ib(default="epsg:4326", validator=[attr.validators.instance_of(str)])


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


class RasterData:
    """
    This class allows to simply create data to unet model, convolutional neural network and ANN network.
    Data is prepared based on two arrays, they have to have equal shape

    """

    def __init__(self, array: np.ndarray, label_array: np.ndarray):
        # self.assert_equal_size(array, label_array)
        self.__array = array
        self.__label_array = label_array

    def prepare_unet_images(self, image_size: List[int]):
        x_shape = self.__label_array.shape[1]
        y_shape = self.__label_array.shape[0]
        for curr_x_shape in range(0, y_shape, image_size[0]):
            for curr_y_shape in range(0, x_shape, image_size[1]):
                main_image_sample = self.__array[curr_y_shape: curr_y_shape + image_size[0],
                                                 curr_x_shape: curr_x_shape + image_size[1],
                                                 :]
                label_image_sample = self.__label_array[curr_y_shape: curr_y_shape + image_size[0],
                                                        curr_x_shape: curr_x_shape + image_size[1],
                                                        :]
                if np.unique(label_image_sample).__len__() > 1:

                    yield [main_image_sample, label_image_sample]

    def save_unet_images(self, image_size: List[int], out_put_path: str):
        for index, (image, label) in enumerate(self.prepare_unet_images(image_size)):
            pixel = Pixel(10, 10)
            current_label_image = Raster.from_array(label, pixel)
            current_image = Raster.from_array(image, pixel)
            try:
                current_label_image.save_gtiff(out_put_path + f"/label/label_{index}.tif", gdal.GDT_Byte)
                current_image.save_gtiff(out_put_path + f"/image/image_{index}.tif", gdal.GDT_Int16)
            except Exception as e:
                logger.error(e)

    def save_con_images(self):
        pass

    def save_ann_csv(self):
        pass

    def assert_equal_size(self, array1: np.ndarray, array2: np.ndarray):
        shape_1 = ArrayShape(array1.shape)
        shape_2 = ArrayShape(array2.shape)
        try:
            assert shape_1 == shape_2
        except AssertionError:
            raise ValueError("Arrays dont have the same size")

