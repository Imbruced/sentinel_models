from typing import List

import attr

from exceptions import PixelSizeException
from exceptions.exceptions import DimensionException
from gis.raster_components import ArrayShape
from preprocessing.data_creators import UnetDataCreator, AnnDataCreator
from preprocessing.data_preparation import ModelData


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

    def prepare_unet_data(self, image_size: List[int], remove_empty_labels=True, threshold=0.0) -> ModelData:
        return UnetDataCreator(
            image=self.image,
            label=self.label,
            image_size=image_size,
            remove_empty_labels=remove_empty_labels,
            threshold=threshold
        ).create()

    def prepare_ann_data(self) -> ModelData:
        return AnnDataCreator(
                    image=self.image,
                    label=self.label,
                ).create()

    def prepare_cnn_data(self, image_size: List[int], remove_empty_labels=True, threshold=0.0):
        pass

    def assert_equal_size(self):
        shape_1 = ArrayShape(self.label.array.shape[:2])
        shape_2 = ArrayShape(self.image.array.shape[:2])

        try:
            assert shape_1 == shape_2
        except AssertionError:
            raise ValueError("Arrays do not have the same size")
