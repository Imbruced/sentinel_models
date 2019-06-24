from abc import ABC
from typing import List
import os
from functools import reduce

import attr
import numpy as np
import gdal
import pandas as pd
from sklearn.model_selection import train_test_split

from exceptions.exceptions import PixelSizeException, DimensionException
from gis import Raster
from gis.raster_components import create_two_dim_chunk
from gis.raster_components import ArrayShape
from logs import logger
from gis import Pixel


@attr.s
class ModelData(ABC):
    x = attr.ib(type=List[Raster])
    y = attr.ib(type=List[Raster])
    test_size = attr.ib(default=0.15)
    random_state = attr.ib(default=2018)

    def __attrs_post_init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.number_of_classes = np.unique(self.y)


@attr.s
class UnetData(ModelData):
    """
    TODO
    add validators
    """
    test_size = attr.ib(default=0.1)
    random_state = attr.ib(default=2018)


# @attr.s
# class UnetConfig(ModelConfig):
#
#     input_size = attr.ib(default=(128, 128, 13))
#     filters = attr.ib(default=16)


def image_meet_pixel_criteria_based_on_threshold(label_image: np.array, image_size: List[int], threshold: float = 30):
    number_of_pixels = reduce((lambda x, y: x * y), image_size)

    uniques, counts = np.unique(label_image, return_counts=True)
    minimum_counts = min(np.unique(label_image, return_counts=True)[1].tolist())
    percent_of_minimum = float(minimum_counts) / float(number_of_pixels) * 100.0
    unique_values = uniques.tolist().__len__()

    return unique_values > 1 and percent_of_minimum > threshold


def image_meets_criteria(*args):
    return True


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
        self.assert_equal_size()
        try:
            assert self.label.extent == self.image.extent
        except AssertionError:
            raise DimensionException("Images does not have the same extents")

    def prepare_unet_images(self, image_size: List[int], remove_empty_labels=True, threshold=30.0):
        """TODO assign proper extents"""

        chunks_array = create_two_dim_chunk(self.image, image_size)
        chunks_label = create_two_dim_chunk(self.label, image_size)
        x_data = []
        y_data = []

        removal_f = image_meet_pixel_criteria_based_on_threshold if remove_empty_labels else image_meets_criteria
        extents = self.image.extent.divide(image_size[0]*self.image.pixel.x, image_size[0]*self.image.pixel.y)

        for img, lbl, extent in zip(chunks_array, chunks_label, extents):
            print(extent.to_wkt())
            if removal_f(lbl, image_size, threshold):
                x_data.append(Raster.from_array(img, img.pixel, extent))
                y_data.append(Raster.from_array(lbl, img.pixel, extent))
        return UnetData(x_data, y_data)

    def assert_equal_size(self):
        shape_1 = ArrayShape(self.label.array.shape[:2])
        shape_2 = ArrayShape(self.image.array.shape[:2])

        try:
            assert shape_1 == shape_2
        except AssertionError:
            raise ValueError("Arrays dont have the same size")




    # def save_con_images(self, path: str, image_shape):
    #     i_shape, j_shape = self.image.array.shape[:2]
    #     pixel = self.image.pixel
    #     for i in range(image_shape-1, i_shape-image_shape):
    #         for j in range(image_shape-1, j_shape-image_shape):
    #             value = self.label.array[i, j, 0]
    #             current_array = self.image.array[i:i+image_shape, j:j+image_shape, :]
    #             logger.info(current_array.shape)
    #             image = Raster.from_array(current_array, pixel)
    #             current_path = os.path.join(path, f"class_{value}")
    #             if not os.path.exists(current_path):
    #                 os.mkdir(current_path)
    #             if value != 0:
    #                 image.save_gtiff(os.path.join(current_path, f"{str(j_shape*i + j)}.tif"), gdal.GDT_Int16)
    #
    # def create_ann_frame(self) -> pd.DataFrame:
    #     image_records = self.image.array.shape[0] * self.image.array.shape[1]
    #     image_dim = self.image.array.shape[-1]
    #
    #     image_melted = self.image.array.reshape(image_records, image_dim)
    #     label_melted = self.label.array.reshape(image_records, 1)
    #
    #     merged_data = np.concatenate([image_melted, label_melted], axis=1)
    #     merged_data_df = pd.DataFrame(merged_data, columns=[*list(range(image_dim)),
    #                                                         "label"])
    #
    #     return merged_data_df[merged_data_df["label"]!=0]