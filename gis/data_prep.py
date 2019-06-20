from typing import List
import os
from functools import reduce

import attr
import numpy as np
import gdal
import pandas as pd

from exceptions.exceptions import PixelSizeException
from gis.raster_components import create_two_dim_chunk
from gis.raster import Raster
from gis.raster_components import ArrayShape
from logs.log_lib import logger
from models import UnetData


@attr.s
class RasterData:
    """
    This class allows to simply create data to unet model, convolutional neural network and ANN network.
    Data is prepared based on two arrays, they have to have equal shape
    TODO Add asserting methods
    """

    image = attr.ib()
    label = attr.ib()

    def __attrs_post_init__(self):
        if self.image.pixel != self.label.pixel:
            raise PixelSizeException("Label and array pixel has to be the same in size")

        # self.assert_equal_size(self.image.array, self.label.array)

    def _with_empty_removal(self, label_image: np.array, image_size: List[int]):
        number_of_pixels = reduce((lambda x, y: x*y), image_size)

        uniques, counts = np.unique(label_image, return_counts=True)
        minimum_counts = min(np.unique(label_image, return_counts=True)[1].tolist())
        percent_of_minimum = float(minimum_counts)/float(number_of_pixels) * 100.0
        unique_values = uniques.tolist().__len__()

        return unique_values > 1 and percent_of_minimum > 30.0

    def _without_empty_romval(self, label_image: np.array, image_size: List[int]):
        return True

    def prepare_unet_images(self, image_size: List[int], remove_empty_labels=True):
        chunks_array = create_two_dim_chunk(self.image.array, image_size)
        chunks_label = create_two_dim_chunk(self.label.array, image_size)
        x_data = []
        y_data = []

        asigning_function = self._with_empty_removal if remove_empty_labels else self._without_empty_romval

        for img, lbl in zip(chunks_array, chunks_label):
            if asigning_function(lbl, image_size):
                x_data.append(img)
                y_data.append(lbl)

        logger.info(np.array(x_data).shape)
        logger.info(np.array(y_data).shape)
        return UnetData(np.array(x_data), np.array(y_data))

    def save_unet_images(self, image_size: List[int], out_put_path: str):
        for index, (image, label) in enumerate(self.prepare_unet_images(image_size)):
            logger.info(image.shape)
            current_label_image = Raster.from_array(label, self.image.pixel)
            current_image = Raster.from_array(image, self.label.pixel)
            try:
                current_label_image.save_gtiff(out_put_path + f"/label/label_{index}.tif", gdal.GDT_Byte)
                current_image.save_gtiff(out_put_path + f"/image/image_{index}.tif", gdal.GDT_Int16)
            except Exception as e:
                logger.error(e)

    @classmethod
    def from_path(cls, path: str, label_name: str, image_name: str):
        label_path = os.path.join(path, label_name)
        image_path = os.path.join(path, image_name)
        pass

    def save_con_images(self, path: str, image_shape):
        i_shape, j_shape = self.image.array.shape[:2]
        pixel = self.image.pixel
        for i in range(image_shape-1, i_shape-image_shape):
            for j in range(image_shape-1, j_shape-image_shape):
                value = self.label.array[i, j, 0]
                current_array = self.image.array[i:i+image_shape, j:j+image_shape, :]
                logger.info(current_array.shape)
                image = Raster.from_array(current_array, pixel)
                current_path = os.path.join(path, f"class_{value}")
                if not os.path.exists(current_path):
                    os.mkdir(current_path)
                if value != 0:
                    image.save_gtiff(os.path.join(current_path, f"{str(j_shape*i + j)}.tif"), gdal.GDT_Int16)

    def create_ann_frame(self) -> pd.DataFrame:
        image_records = self.image.array.shape[0] * self.image.array.shape[1]
        image_dim = self.image.array.shape[-1]

        image_melted = self.image.array.reshape(image_records, image_dim)
        label_melted = self.label.array.reshape(image_records, 1)

        merged_data = np.concatenate([image_melted, label_melted], axis=1)
        merged_data_df = pd.DataFrame(merged_data, columns=[*list(range(image_dim)),
                                                            "label"])

        return merged_data_df[merged_data_df["label"]!=0]

    def assert_equal_size(self, array1: np.ndarray, array2: np.ndarray):
        shape_1 = ArrayShape(array1.shape)
        shape_2 = ArrayShape(array2.shape)
        try:
            assert shape_1 == shape_2
        except AssertionError:
            raise ValueError("Arrays dont have the same size")
