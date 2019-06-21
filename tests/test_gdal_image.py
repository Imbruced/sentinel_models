import os
from unittest import TestCase

import gdal
import numpy as np

from exceptions.exceptions import FormatNotAvailable
from gis.geometry import Origin
from gis.image_data import GdalImage, Raster, ImageWriter, ImageReader
from gis.io_abstract import ClsFinder, DefaultOptionRead
from gis.raster_components import Pixel
from logs import logger
from exceptions import OptionNotAvailableException
from plotting import ImagePlot

TEST_IMAGE_PATH = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\pictures\\buildings.tif"


class TestImageDataModule(TestCase):
    cls_finder = ClsFinder(__name__)
    empty_raster = Raster.empty()
    image_writer = ImageWriter(data=empty_raster)
    path = os.path.join(os.getcwd(), "test_data")

    def test_gdal_image_from_file(self):
        gd = GdalImage.load_from_file(TEST_IMAGE_PATH, "epsg:4326")
        self.assertTrue(isinstance(gd.array, np.ndarray))

    def test_gdal_image_from_file_image_accuracy(self):
        gd = GdalImage.load_from_file(TEST_IMAGE_PATH, "epsg:4326")
        array = gd.array
        self.assertEqual(array[array == 255].__len__(), 25121)
        self.assertEqual(array.shape, (3, 1677, 1673))

    def test_gdal_image_to_raster(self):
        gd = GdalImage.load_from_file(TEST_IMAGE_PATH, "epsg:4326")
        self.assertEqual(isinstance(gd.to_raster(), Raster), True)

    def test_insert_polygon(self):
        gdal_in_memory = GdalImage.in_memory(100, 100)
        gdal_in_memory.transform(Origin(
            100.0, 100.0
        ),
            Pixel(1.0, 1.0)
        )
        gdal_in_memory.insert_polygon(
            "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))",
            10
        )
        array = gdal_in_memory.array

        self.assertEqual(array[array == 10].size, 121)

    def test_cls_names(self):
        logger.info(self.cls_finder.available_cls)

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
        file_path = os.path.join(self.path, "output.tif")
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

    def test_convertion_to_raster(self):

        gdal_in_memory = GdalImage.in_memory(100, 100)
        gdal_in_memory.transform(Origin(
            100.0, 100.0
        ),
            Pixel(1.0, 1.0)
        )
        gdal_in_memory.insert_polygon(
            "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))",
            10
        )
        raster = gdal_in_memory.to_raster()

        return raster

    def test_save_to_geotiff(self):

        TEST_FILE = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\pictures\\test.tif"
        try:
            os.remove(TEST_FILE)
        except Exception as e:
            pass
        raster = self.test_convertion_to_raster()
        raster. \
            write. \
            format("geotiff"). \
            options(dtype=gdal.GDT_Float32). \
            save(TEST_FILE)

    def test_save_to_png(self):

        TEST_FILE = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\pictures\\test.png"
        try:
            os.remove(TEST_FILE)
        except Exception as e:
            pass
        raster = self.test_convertion_to_raster()
        raster. \
            write. \
            format("png"). \
            save(TEST_FILE)

    def test_reading_geotif(self):
        raster = Raster\
            .read\
            .format("geotiff")\
            .load(TEST_IMAGE_PATH)

        array = raster.array
        self.assertEqual(array[array == 255].__len__(), 25121)
        self.assertEqual(array.shape, (1677, 1673, 3))

    def test_reading_raster_from_wkt(self):
        wkt = "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))"

        raster = Raster \
            .read \
            .format("wkt") \
            .options(
             pixel=Pixel(0.2, 0.2),
             value=10
            ).load(wkt)

        array = raster.array
        self.assertEqual(array[array == 10].size, 2601)

    def test_raster_show(self):
        wkt = "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))"

        raster = Raster \
            .read \
            .format("wkt") \
            .options(
                pixel=Pixel(0.7, 0.7),
                value=10
             ).load(wkt)
        # raster.show()

    def test_reading_raster_from_shapefile(self):
        shape_path = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\shapes\\domy.shp"

        raster = Raster \
            .read \
            .format("shp") \
            .load(shape_path)
        raster.show()

