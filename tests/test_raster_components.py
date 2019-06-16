import os
import sys
from unittest import TestCase

import gdal

from exceptions import OptionNotAvailableException
from gis.raster import Raster
from gis.writers import ImageWriter, ClsFinder
from logs.log_lib import logger


class TestClsFinder(TestCase):
    cls_finder = ClsFinder(__name__)

    def test_cls_names(self):
        logger.info(self.cls_finder.available_cls)


class TestImageOptions(TestCase):
    pass


class TestImageWriter(TestCase):
    empty_raster = Raster.empty_raster()
    image_writer = ImageWriter(empty_raster)
    path = os.path.join(os.getcwd(), "test_data")

    def test_writers(self):
        self.assertGreater(
            self.image_writer.writers.__len__(),
            1,
            self.image_writer.writers
        )
        logger.info(self.image_writer.writers)

    def test_options_keys(self):
        self.image_writer.options(format="png")
        with self.assertRaises(OptionNotAvailableException):
            self.image_writer.options(random_key=10)
            self.image_writer.options(another_random_key=10)

    def test_options_values(self):
        with self.assertRaises(OptionNotAvailableException):
            self.image_writer.options(format="gtp").save("empty_path")

    def test_available_format(self):
        self.image_writer.options(format="geotiff")

    def test_save_png(self):
        file_path = os.path.join(self.path, "output.png")
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        self.image_writer.options(format="png").save(file_path)

    def test_save_geotiff(self):
        file_path = os.path.join(self.path, "output.geotiff")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        self.image_writer.options(format="geotiff", dtype=gdal.GDT_Byte).save(file_path)




