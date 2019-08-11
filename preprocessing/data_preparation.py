from abc import ABC
from typing import List
from functools import reduce

import attr
import numpy as np
from sklearn.model_selection import train_test_split

from exceptions.exceptions import PixelSizeException, DimensionException
from gis import Raster
from gis.raster_components import create_two_dim_chunk
from gis.raster_components import ArrayShape
from utils.decorators import lazy_property


@attr.s
class ModelData(ABC):
    x = attr.ib(type=List['r.Raster'])
    y = attr.ib(type=List['r.Raster'])
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


def image_meet_pixel_criteria_based_on_threshold(label_image: np.array, image_size: List[int], threshold: float = 30):
    number_of_pixels = reduce((lambda x, y: x * y), image_size)

    uniques, counts = np.unique(label_image, return_counts=True)
    minimum_counts = min(np.unique(label_image, return_counts=True)[1].tolist())
    percent_of_minimum = float(minimum_counts) / float(number_of_pixels) * 100.0
    unique_values = uniques.tolist().__len__()

    return unique_values > 1 and percent_of_minimum > threshold


def image_meets_criteria(*args):
    return True


class AnnData(ModelData):

    test_size = attr.ib(default=0.1)
    random_state = attr.ib(default=2018)


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

    def prepare_unet_images(self, image_size: List[int], remove_empty_labels=True, threshold=0.0):
        return UnetDataCreator(
            image=self.image,
            label=self.label,
            image_size=image_size,
            remove_empty_labels=remove_empty_labels,
            threshold=threshold
        ).create()

    def prepare_ann_data(self):
        pass

    def prepare_cnn_data(self):
        pass

    def assert_equal_size(self):
        shape_1 = ArrayShape(self.label.array.shape[:2])
        shape_2 = ArrayShape(self.image.array.shape[:2])

        try:
            assert shape_1 == shape_2
        except AssertionError:
            raise ValueError("Arrays do not have the same size")


@attr.s
class DataCreator(ABC):

    image = attr.ib(type='r.Raster', validator=[])
    label = attr.ib(type='r.Raster', validator=[])

    def __attrs_post_init__(self):
        """
        TODO Validate dimension inequality
        TODO deimension validator for image and label
        :return:
        """
        pass

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
        """TODO assign proper extents"""
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
    def __wide_data(raster: 'r.Raster') -> np.ndarray:
        height, width, dim = raster.shape
        return raster.reshape(height*width, dim)
