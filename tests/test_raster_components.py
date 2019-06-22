import os
from unittest import TestCase

import gdal

from exceptions import OptionNotAvailableException
from exceptions.exceptions import FormatNotAvailable
from gis.io_abstract import DefaultOptionRead
from gis.image_data import Raster
from gis.image_data import ImageReader
from gis.image_data import ImageWriter
from logs.log_lib import logger


TEST_IMAGE_PATH = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\pictures\\buildings.tif"


class TestImageWriter(TestCase):
    empty_raster = Raster.empty()
    image_writer = ImageWriter(data=empty_raster)
    path = os.path.join(os.getcwd(), "test_data")

    def test_writers(self):
        self.assertGreater(
            self.image_writer.writers.__len__(),
            1,
            self.image_writer.writers
        )
        logger.info(self.image_writer.writers)

    def test_options_keys(self):
        self.image_writer.format("png")
        with self.assertRaises(OptionNotAvailableException):
            self.image_writer.options(random_key=10)
            self.image_writer.options(another_random_key=10)

    def test_options_values(self):
        with self.assertRaises(FormatNotAvailable):
            self.image_writer.format("gtp").save("empty_path")

    def test_available_format(self):
        self.image_writer.format("geotiff")

    def test_save_png(self):
        file_path = os.path.join(self.path, "output.png")
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        self.image_writer.format("png").save(file_path)

    def test_save_geotiff(self):
        file_path = os.path.join(self.path, "output.geotiff")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        self.image_writer.format("geotiff").options(dtype=gdal.GDT_Byte).save(file_path)

    def test_format_image_reader(self):
        reader = ImageReader().format("png")
        self.assertEqual(reader.io_options, DefaultOptionRead.png())

    def test_format_in_options(self):
        with self.assertRaises(AttributeError):
            self.image_writer.options(format="png")

    def test_loading_geotiff(self):
        image_reader = ImageReader().format("geotiff").load(
            TEST_IMAGE_PATH
        )



    # def test_readers(self):
    #     img_reader = ImageReader(
    #         io_options=Options(delimiter="csv")
    #     )




