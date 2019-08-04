from unittest import TestCase

import numpy as np

from gis import raster as r
from gis import geometry as g
from gis import raster_components as rc
from gis.crs import Crs
from preprocessing.data_preparation import AnnDataCreator

TEST_IMAGE_PATH = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\pictures\\buildings.tif"


class TestAnnPreparation(TestCase):
    empty_array_image = np.array([
        [[1, 2, 3], [5, 4, 7], [9, 3, 5]],
        [[5, 4, 7], [5, 4, 7], [9, 3, 5]],
        [[9, 3, 5], [5, 4, 7], [9, 3, 5]]
    ])
    empty_array_label = np.array([
        [3, 20, 4],
        [11, 5, 10],
        [20, 5, 4]
    ]).reshape(3, 3, 1)
    image = r.Raster.from_array(
        empty_array_image,
        pixel=rc.Pixel(1.0, 1.0),
        extent=g.Extent.from_coordinates([0, 0, 1, 1], crs=Crs(epsg="epsg:2180"))
    )
    label = r.Raster.from_array(
        empty_array_label,
        pixel=image.pixel,
        extent=image.extent
    )

    ann_data = AnnDataCreator(
        label=empty_array_label,
        image=empty_array_image
    )

    def test_label_wide(self):
        target_res = np.array([[3], [20], [4], [11], [5], [10], [20], [5], [4]])
        wide_label = self.ann_data.wide_label
        self.assertEqual(np.array_equal(wide_label, target_res), True)

    def test_image_wide(self):
        target_res = np.array([
            [1, 2, 3], [5, 4, 7], [9, 3, 5],
            [5, 4, 7], [5, 4, 7], [9, 3, 5],
            [9, 3, 5], [5, 4, 7], [9, 3, 5]
        ])
        wide_image = self.ann_data.wide_image
        self.assertEqual(np.array_equal(wide_image, target_res), True)

    def test_concat_arrays(self):
        target_res = np.array(
            [
                [1, 2, 3, 3],
                [5, 4, 7, 20],
                [9, 3, 5, 4],
                [5, 4, 7, 11],
                [5, 4, 7, 5],
                [9, 3, 5, 10],
                [9, 3, 5, 20],
                [5, 4, 7, 5],
                [9, 3, 5, 4]
            ]
        )
        concat_ = self.ann_data.concat_arrays()
        self.assertEqual(np.array_equal(concat_, target_res), True)

    def test_create(self):
        target_test_x = np.array([
            [5, 4, 7],
            [9, 3, 5]
        ])
        target_test_y = np.array([
            [11],
            [4]
        ])
        self.assertEqual(np.array_equal(target_test_x, self.ann_data.create().x_test), True)
        self.assertEqual(np.array_equal(target_test_y, self.ann_data.create().y_test), True)
