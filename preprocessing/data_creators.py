from abc import ABC
from functools import reduce
from typing import List, Tuple

import attr
import numpy as np

from exceptions.exceptions import SizeException
from gis import Raster, Extent
from gis.raster_components import create_two_dim_chunk, ReferencedArray
from logs import logger
from preprocessing.data_preparation import ModelData, UnetData, AnnData, CnnData
from utils.decorators import lazy_property


def image_meet_pixel_criteria_based_on_threshold(label_image: Raster, image_size: List[int], threshold: float = 30.0):
    number_of_pixels = reduce((lambda x, y: x * y), image_size)

    uniques, counts = np.unique(label_image, return_counts=True)
    minimum_counts = min(np.unique(label_image, return_counts=True)[1].tolist())
    percent_of_minimum = float(minimum_counts) / float(number_of_pixels) * 100.0
    unique_values = uniques.tolist().__len__()

    return unique_values > 1 and percent_of_minimum > threshold


def image_meets_criteria(*args):
    return True


def check_image_type(image: Raster) -> bool:
    return type(image) == Raster


def check_size(size: int) -> bool:
    return size % 2 == 1 and type(size) == int


def check_parameters(image: Raster, size: int) -> bool:
    return check_image_type(image) and check_size(size)


def split_images_to_cnn(image: Raster, window_size: int) -> Tuple[List[Raster], int]:
    """TODO create generator"""
    check_parameters(image=image, size=window_size)
    image_size_x, image_size_y = image.shape[:2]
    dx = int(window_size/2)
    staring_index_x, ending_index_x = dx, image_size_x - dx-1
    staring_index_y, ending_index_y = dx, image_size_y - dx-1
    if ending_index_x < staring_index_x or ending_index_y < staring_index_y:
        raise SizeException("Window shape is to huge")

    pixel_size_x = image.pixel.x
    pixel_size_y = image.pixel.y
    extent_shape_y = image.extent.dy
    data = []
    left_up = image.extent.left_down.translate(0, extent_shape_y)

    for y in range(staring_index_y, ending_index_y + 1):
        for x in range(staring_index_x, ending_index_x + 1):
            current_extent = Extent.from_coordinates(
                coordinates=[
                    left_up.x + (x * pixel_size_x),
                    left_up.y - ((y+1) * abs(pixel_size_y)),
                    left_up.x + ((x+1) * pixel_size_x),
                    left_up.y - (y * abs(pixel_size_y)),

                ],
                crs=image.crs
            )
            raster = Raster.from_array(
                array=image.array[x-dx: x + dx + 1, y-dx: y + dx + 1],
                extent=current_extent,
                pixel=image.pixel
            )
            data.append(raster)

    return data, data.__len__()


@attr.s
class DataCreator(ABC):
    image = attr.ib(type=Raster, validator=[])
    label = attr.ib(type=Raster, validator=[])

    def create(self) -> ModelData:
        raise NotImplementedError()


@attr.s
class UnetDataCreator(DataCreator):
    image_size = attr.ib(type=List[int])
    remove_empty_labels = attr.ib(type=bool, default=True)
    threshold = attr.ib(default=0.0)

    def create(self) -> ModelData:
        return self.prepare_images()

    def prepare_images(self):
        x_data = []
        y_data = []

        for img, lbl, extent in zip(self.chunks_array, self.chunks_label, self.extents):
            if self.removal_function(lbl, self.image_size, self.threshold):
                x_data.append(Raster.from_array(img, img.pixel, extent))
                y_data.append(Raster.from_array(lbl, img.pixel, extent))
        return UnetData(x_data, y_data)

    @lazy_property
    def extents(self):
        return self.create_extents()

    @lazy_property
    def chunks_label(self):
        return self.chunks[1]

    @lazy_property
    def chunks_array(self):
        return self.chunks[0]

    @lazy_property
    def chunks(self):
        chunks_array = create_two_dim_chunk(self.image, self.image_size)
        chunks_label = create_two_dim_chunk(self.label, self.image_size)

        return chunks_array, chunks_label

    def create_extents(self):
        return self.image.extent.divide(
            self.image_size[0] * self.image.pixel.x,
            self.image_size[0] * self.image.pixel.y
        )

    @property
    def removal_function(self):
        return image_meet_pixel_criteria_based_on_threshold if self.remove_empty_labels else image_meets_criteria


@attr.s
class AnnDataCreator(DataCreator):

    def create(self) -> ModelData:
        return AnnData(
            x=self.wide_image,
            y=self.wide_label
        )

    def concat_arrays(self) -> np.ndarray:
        wide_label = self.wide_label
        wide_image = self.wide_image
        return np.concatenate([wide_image, wide_label], axis=1)

    @lazy_property
    def wide_label(self) -> np.ndarray:
        return self.__wide_label_data()

    @lazy_property
    def wide_image(self) -> np.ndarray:
        return self.__wide_image_data()

    def __wide_label_data(self) -> np.ndarray:
        return self.__wide_data(self.label)

    def __wide_image_data(self) -> np.ndarray:
        return self.__wide_data(self.image)

    @staticmethod
    def __wide_data(raster: Raster) -> np.ndarray:
        height, width, dim = raster.shape
        return raster.reshape(height*width, dim)


@attr.s
class CnnDataCreator(DataCreator):
    window_size = attr.ib(type=int)

    def create(self) -> CnnData:
        images, labels = self.prepare(self.window_size)
        return CnnData(
            images,
            labels
        )

    def prepare(self, window_size: int) -> Tuple[List[Raster], List[int]]:
        # labels = split_images_to_cnn(self.label, window_size)
        # labels = self.label.reshape(self.label.shape[0]*self.label.shape[1])
        images, labels = split_images_to_cnn(self.image, window_size)

        return images, [el for el in range(labels)]
