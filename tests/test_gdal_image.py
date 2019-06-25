import os
from unittest import TestCase

import gdal
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

from exceptions.exceptions import FormatNotAvailable, CrsException
from gis.data_prep import RasterData
from gis.geometry import Origin, Wkt, Point, Extent, GeometryFrame, Polygon, InteractiveGeometryPlotter
from gis.image_data import GdalImage, Raster, ImageWriter, ImageReader
from gis.io_abstract import ClsFinder, DefaultOptionRead
from gis.raster_components import Pixel
from gis.crs import Crs
from gis.image_data import ImageStand
from logs import logger
from exceptions import OptionNotAvailableException
from models import Unet, UnetConfig
from plotting import SubPlots

TEST_IMAGE_PATH = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\pictures\\buildings.tif"


class TestImageDataModule(TestCase):
    cls_finder = ClsFinder(__name__)
    empty_raster = Raster.empty()
    image_writer = ImageWriter(data=empty_raster)
    path = os.path.join(os.getcwd(), "test_data")

    POLYGON = "Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))"
    LINESTRING = "LINESTRING(-71.160281 42.258729,-71.160837 42.259113,-71.161144 42.25932)"
    POINT = "POINT(-71 42)"
    shape_path = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\data\\shapes\\domy.shp"

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

    def test_reading_raster_from_shapefile_value_baased_on_column(self):
        raster = Raster \
            .read \
            .format("shp") \
            .options(color_column="cls") \
            .load(self.shape_path)
        self.assertEqual(np.unique(raster.array).tolist(), [0, 1, 2, 3, 4, 5])

        # raster.show()

    def test_reading_raster_from_shapefile_value_the_same(self):
        raster = Raster \
            .read \
            .format("shp") \
            .load(self.shape_path)
        self.assertEqual(np.unique(raster.array).tolist(), [0, 1])
        # raster.show()

    def test_reading_raster_from_shapefile_all_values_different(self):

        raster = Raster \
            .read \
            .format("shp") \
            .options(all_unique="True") \
            .load(self.shape_path)
        self.assertEqual(np.unique(raster.array).tolist(), list(range(0, 90)))

    def test_load_bigger_shapefile_test_exception(self):
        path = "D:\\master_thesis\\data\\geometry\\features\\crops.shp"
        with self.assertRaises(ValueError):
            Raster \
                .read \
                .format("shp") \
                .load(path) \
                .show()

    def test_wkt_instance(self):
        wkt = Wkt("Polygon((110.0 110.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 110.0))")

    def test_split_coordinates(self):
        wkt1 = Wkt(self.POLYGON)
        wkt2 = Wkt(self.POINT)
        wkt3 = Wkt(self.LINESTRING)

        self.assertEqual(
            wkt1.coordinates,
            [
                Point(x[0], x[1]) for x in
                [[110.0, 110.0], [110.0, 120.0], [120.0, 120.0], [120.0, 110.0], [110.0, 110.0]]
             ]
        )
        self.assertEqual(
            wkt2.coordinates,
            [Point(-71.0, 42.0)]
        )

        self.assertEqual(
            wkt3.coordinates,
            [
                Point(x[0], x[1]) for x in
                [[-71.160281, 42.258729], [-71.160837, 42.259113], [-71.161144, 42.25932]]
            ]
        )

    def test_extent_wkt(self):
        wkt1 = Wkt(self.POLYGON)
        self.assertEqual(
            wkt1.extent,
            Extent(Point(x=110.0, y=110.0), Point(x=120.0, y=120.0))
        )

    def test_showing_geoframe_like_spark(self):

        frame = GeometryFrame.from_file(self.shape_path).show(2, True)

    def test_extent_to_wkt(self):
        extent = Extent()
        self.assertEqual(extent.to_wkt(), "POLYGON((0.0 0.0, 0.0 1.0, 1.0 1.0, 1.0 0.0, 0.0 0.0))")

    def test_point(self):
        p = Point(21.0, 52.0)

    def test_polygon(self):
        coordinates = [[110.0, 110.0], [110.0, 120.0], [120.0, 120.0], [120.0, 110.0], [110.0, 110.0]]
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(coordinates)

    def test_polygon_from_wkt(self):
        poly = Polygon.from_wkt("POLYGON((0.0 0.0, 0.0 1.0, 1.0 1.0, 1.0 0.0, 0.0 0.0))", Crs("local"))
        self.assertEqual(poly.area, 1.0)

    def test_interactive_plot(self):
        gdf = GeometryFrame.from_file(self.shape_path)
        InteractiveGeometryPlotter(gdf).plot()

    def test_gdf_interactive_plot(self):
        gdf = GeometryFrame.from_file(self.shape_path)
        gdf.plot(interactive=True)

    def test_extent_comparison(self):
        extent1 = Extent.from_coordinates([100.0, 100.0, 120.0, 120.0], crs=Crs("epsg:4326"))
        extent2 = Extent.from_coordinates([100.00, 100.00, 120.00, 120.00], crs=Crs("epsg:4326"))
        assert extent1 == extent2

    def test_geometry_frame_from_file(self):
        gdf = GeometryFrame.from_file(self.shape_path)
        self.assertEqual(gdf.crs, Crs("epsg:26917"))

    def test_different_crs_and_extent_crs(self):

        with self.assertRaises(CrsException):

            image = Raster \
                .read \
                .options(crs=Crs("epsg:4326"))\
                .format("geotiff") \
                .load(TEST_IMAGE_PATH)

            label = Raster \
                .read \
                .format("shp") \
                .options(
                    extent=image.extent,
                    crs=Crs("epsg:2008"),
                    pixel=image.pixel
                ) \
                .load(self.shape_path)

    def test_creating_unet_images(self):
        image = Raster \
            .read \
            .format("geotiff") \
            .load(TEST_IMAGE_PATH)

        label: Raster = Raster \
            .read \
            .format("shp") \
            .options(
                pixel=image.pixel,
                extent=image.extent
            ) \
            .load(self.shape_path)

        raster_data = RasterData(image=image, label=label)
        images = raster_data.prepare_unet_images([128, 128])
        # print(images.x_train[1].shape)
        images.x[0].write\
            .format("geotiff")\
            .save("C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\image_0.tif")

        images.x[1].write\
            .format("geotiff")\
            .save("C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\image_1.tif")

        images.x[2].write\
            .format("geotiff")\
            .save("C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\image_2.tif")


    def test_extent_split(self):
        extent = Extent.from_coordinates([20.0, 20.0, 50.0, 50.0], "epsg:2180").divide_dy(5.1)
        for ex in extent:
            divide_x = ex.divide_dx(5.1)
            for dvx in divide_x:
                print(dvx.to_wkt())

    def test_image_standarizer(self):
        image = Raster \
            .read \
            .format("geotiff") \
            .load(TEST_IMAGE_PATH)

        standarize1 = ImageStand(raster=image)
        standarized = standarize1.standarize_image(StandardScaler())
        cnt = standarized[(standarized >= 2.73777289) & (standarized <= 3.0)]
        self.assertEqual(cnt.size, 4135)

    def test_unet_model(self):
        image = Raster \
            .read \
            .format("geotiff") \
            .load(TEST_IMAGE_PATH)

        label: Raster = Raster \
            .read \
            .format("shp") \
            .options(
            pixel=image.pixel,
            extent=image.extent
        ) \
            .load(self.shape_path)

        standarize1 = ImageStand(raster=image)
        standarized = standarize1.standarize_image(StandardScaler())
        raster_data = RasterData(standarized, label)
        unet_images = raster_data.prepare_unet_images(image_size=(64, 64))

        callbacks = [
            EarlyStopping(patience=100, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=100, min_lr=0, verbose=1),
            ModelCheckpoint('model_more_class_pixels.h5', verbose=1, save_best_only=True, save_weights_only=False)
        ]
        config = UnetConfig(
            input_size=[64, 64, 3],
            metrics=["accuracy"],
            optimizer=SGD(lr=0.001),
            callbacks=callbacks,
            loss="binary_crossentropy",

        )
        #
        unet = Unet(config=config)
        unet.compile()
        unet.fit(unet_images, epochs=1)
        predicted = unet.predict(x=unet_images.x_test[0], threshold=0.4)
        SubPlots().extend(
            predicted,
            unet_images.x_test[0]
        ).plot(nrows=1)

    def test_show_tru_color_image(self):
        image = Raster \
            .read \
            .format("geotiff") \
            .options(crs=Crs("epsg:4326")) \
            .load(
            "C:\\Users\\Pawel\\Downloads\\dstl-satellite-imagery-feature-detection\\three_band\\three_band\\6010_1_2.tif")
        image.show(true_color=True)

