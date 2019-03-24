import numpy as np
from osgeo import osr
import gdal
import ogr
from gis.geometry import Point
import matplotlib.pyplot as plt
from copy import copy
import json
from gis.descriptors import PositiveValue
import matplotlib
import geopandas as gpd
import os
from geopandas import GeoDataFrame
from gis.geometry import GeometryFrame
from gis.descriptors import NumberType
from gis.datum import Crs
from gis.geometry import Origin
from typing import List
from gis.descriptors import PositiveInteger
from gis.geometry import PolygonFrame
import attr
from gis.log_lib import logger


class StanarizeParams:

    _band_number = PositiveInteger()
    _coefficients = {}

    def __init__(self, band_number):
        self._band_number = band_number

    def add(self, coeff, band_name):
        if len(self._coefficients.keys()) >= self._band_number:
            raise OverflowError("Value is to high")
        else:
            try:
                self._coefficients.get(band_name)
            except KeyError:
                self._coefficients[band_name] = coeff

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        raise ValueError("Value can not be set, please use add method")


class ImageStand:
    """
    Class is responsible for scaling image data, it requires gis.raster.Raster object
    Band names can be passed, by default is ["band_index" ... "band_index+n"]
    use standarize_image method to standarize data, it will return Raster with proper refactored image values

    """

    def __init__(self, raster, names=None):
        self.__stan_params = StanarizeParams(raster.array.shape[2])
        self.__names = [f"band_{band}" for band in range(raster.array.shape[2])] if names is None else names
        logger.info(self.__names)
        self.__raster = raster

    def save_params(self, path):
        if os.path.exists(path):
            raise FileExistsError("File with this name exists in directory")
        with open(path, "w") as file:
            params_json = json.load(self.__stan_params.coefficients)
            file.writelines(params_json)

    def standarize_image(self, rescale_function=None):
        """
        Rescale data by maximum
        :return:
        """
        empty_array = []

        for index, band in enumerate(self.__names):
            array_divided = self.__stand_one_dim(self.__raster.array[:, :, index], band, rescale_function)
            empty_array.append(array_divided)
            array_divided = None

        return Raster(self.__raster.extent, self.__raster.pixel, np.array(empty_array).transpose([1, 2, 0]))

    def __stand_one_dim(self, array: np.ndarray, band_name: str, rescale_function=None):
        """
        Standarizing 2 dim array, fnct for rescaling can be passed: require function which takes array and returns
        array
        :param array:
        :return:
        """
        if rescale_function is None:
            stand_value = self.find_max(array)
        else:
            stand_value = rescale_function(array)

        self.__stan_params.add(stand_value, band_name)
        return array/stand_value

    def find_max(self, array):
        return 12000
        # return array.max()

    @property
    def stan_params(self):
        return self.__stan_params

    @stan_params.setter
    def stan_params(self, value):
        raise AttributeError("This param can' t be set")



class Raster:

    __srs = osr.SpatialReference('LOCAL_CS["arbitrary"]')

    def __init__(self, extent: Extent, pixel: Pixel, array: np.ndarray):
        self.__array = array
        self.__extent = extent
        self.__pixel = pixel
        logger.warning(self.__array.shape)

    def save_gtiff(self, path, raster_dtype):
        drv = gdal.GetDriverByName("GTiff")
        if os.path.isfile(path):
            raise FileExistsError("File currently exists")

        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        band_number = self.array.shape[2]

        ds = drv.Create(path, self.array.shape[1], self.array.shape[0], band_number, raster_dtype)
        transformed_ds = self.__transform(ds, self.__extent.origin, self.__pixel)
        transformed_ds.SetProjection(WKT_32364)

        for band in range(self.array.shape[2]):
            transformed_ds.GetRasterBand(band + 1).WriteArray(self.array[:, :, band])

    @staticmethod
    def __create_raster(x_shape: PositiveValue, y_shape: PositiveValue):
        memory_ob = gdal.GetDriverByName('MEM')
        raster = memory_ob.Create('', x_shape, y_shape, 1, gdal.GDT_Byte)

        return raster

    @staticmethod
    def __transform(raster, origin: Origin, pixel: Pixel):
        copy_raster = raster
        copy_raster.SetGeoTransform((origin.x, pixel.x, 0.0, origin.y+(8838*10), 0, -pixel.y))
        left_top_corner_x, pixel_size_x, _, left_top_corner_y, _, pixel_size_y = copy_raster.GetGeoTransform()
        logger.info(left_top_corner_x)
        logger.info(pixel_size_x)
        logger.info(left_top_corner_y)
        logger.info(pixel_size_y)

        copy_raster.SetProjection('LOCAL_CS["arbitrary"]')

        return copy_raster

    @staticmethod
    def __insert_polygon(raster, wkt, value):
        srs = osr.SpatialReference('LOCAL_CS["arbitrary"]')
        rast_ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('wrk')
        rast_mem_lyr = rast_ogr_ds.CreateLayer('poly', srs=srs)
        feat = ogr.Feature(rast_mem_lyr.GetLayerDefn())
        feat.SetGeometryDirectly(ogr.Geometry(wkt=wkt))
        rast_mem_lyr.CreateFeature(feat)
        err = gdal.RasterizeLayer(raster, [1], rast_mem_lyr, None, None, [value], ['ALL_TOUCHED=TRUE'])

        return raster

    @classmethod
    def from_wkt(cls, wkt, extent, pixel):
        """
        TODO simplify this method
        :param wkt:
        :param extent:
        :param pixel:
        :return:
        """

        point_a = Point(0, 0)
        point_b = Point(int(extent.dx / pixel.x), int(extent.dy / pixel.y))
        new_extent = Extent(point_a, point_b)
        raster = cls.__create_raster(new_extent.dx, new_extent.dy)
        transformed_raster = cls.__transform(raster, extent.origin, pixel)
        polygon_within = cls.__insert_polygon(transformed_raster, wkt, 1)

        extent_new = Extent(Point(extent.origin.x, extent.origin.y),
                            Point((new_extent.origin.x + new_extent.dx) * pixel.x,
                                  (new_extent.origin.y + new_extent.dy) * pixel.y))
        array = polygon_within.ReadAsArray()
        raster_ob = cls(extent_new, pixel, array.reshape(*array.shape, 1))

        return raster_ob

    @staticmethod
    def reshape_array(array):
        try:
            th_d = array.shape[2]
            array_copy = array.reshape(*array.shape[1: 3], array.shape[0])
        except ImportError:
            band_number = 1
            array_copy = array.reshape(*array.shape, 1)
        logger.error(f"Wymiar po reshapie {array_copy.shape}")
        return array_copy

    @classmethod
    def from_array(cls, array, pixel, extent=Extent(Point(0, 0), Point(0, 0))):
        array_copy = array

        raster_ob = cls(None, pixel, array_copy)
        return raster_ob

    @classmethod
    def from_geo(cls, geoframe: GeometryFrame, extent: Extent, pixel: Pixel):

        """
        TODO simplify this method
        :param geoframe:
        :param extent:
        :param pixel:
        :return:
        """
        wkt_frame = geoframe.to_wkt()
        wkt_string = wkt_frame.frame["wkt"].values.tolist()[0]

        point_a = Point(0, 0)
        point_b = Point(int(extent.dx / pixel.x), int(extent.dy / pixel.y))
        new_extent = Extent(point_a, point_b)
        raster = cls.__create_raster(new_extent.dx, new_extent.dy)
        transformed_raster = cls.__transform(raster, extent.origin, pixel)

        polygon_within = cls.__insert_polygon(transformed_raster, wkt_string, 1)

        extent_new = Extent(Point(extent.origin.x, extent.origin.y),
                            Point((new_extent.origin.x + new_extent.dx) * pixel.x,
                                  (new_extent.origin.y + new_extent.dy) * pixel.y))
        array = polygon_within.ReadAsArray()
        raster_ob = cls(extent_new, pixel, array.reshape(*array.shape, 1))

        return raster_ob

    @classmethod
    def from_file(cls, path):
        raster_from_file = cls.load_image(path)
        left_top_corner_x, pixel_size_x, _, left_top_corner_y, _, pixel_size_y = raster_from_file.GetGeoTransform()

        pixel_number_x = raster_from_file.RasterXSize
        pixel_number_y = raster_from_file.RasterYSize
        pixel = Pixel(pixel_size_x, -pixel_size_y)

        extent = Extent(Point(left_top_corner_x, left_top_corner_y - -pixel_number_y),
                        Point(left_top_corner_x + pixel_number_x, left_top_corner_y))

        array = cls.reshape_array(cls.gdal_file_to_array(raster_from_file))
        logger.info(array.shape)
        return cls(extent, pixel, array)

    @staticmethod
    def load_image(path):
        ds: gdal.Dataset = gdal.Open(path)
        return ds

    @staticmethod
    def gdal_file_to_array(ds):
        nda = ds.ReadAsArray()

        return nda

    @property
    def array(self) -> np.ndarray:
        return self.__array

    @array.setter
    def array(self, value: np.ndarray):
        self.__array_set += 1
        if self.__array_set > 1:
            logger.error("Value can't be set")
            raise AttributeError("value can't be set")
        self.__array = value

    @property
    def extent(self):
        return self.__extent

    @property
    def pixel(self):
        return self.__pixel


class ArrayShape:
    x_shape = PositiveInteger()
    y_shape = PositiveInteger()

    def __init__(self, shape):
        self.x_shape = shape[0]
        self.y_shape = shape[1]

    def __ne__(self, other):
        return self.x_shape != other.x_shape or self.y_shape != other.y_shape

    def __eq__(self, other):
        return self.x_shape == other.x_shape or self.y_shape == other.y_shape


class RasterData:
    """
    This class allows to simply create data to unet model, convolutional neural network and ANN network.
    Data is prepared based on two arrays, they have to have equal shape

    """

    def __init__(self, array: np.ndarray, label_array: np.ndarray):
        self.assert_equal_size(array, label_array)
        self.__array = array
        self.__label_array = label_array

    def prepare_unet_images(self, image_size: List[int]):
        x_shape = self.__label_array.shape[1]
        y_shape = self.__label_array.shape[0]
        for curr_x_shape in range(0, y_shape, image_size[0]):
            for curr_y_shape in range(0, x_shape, image_size[1]):
                yield [self.__array[curr_y_shape: curr_y_shape + image_size[0],
                       curr_x_shape: curr_x_shape + image_size[1], :],
                       self.__label_array[curr_y_shape: curr_y_shape + image_size[0],
                       curr_x_shape: curr_x_shape + image_size[1], :]]

    def save_unet_images(self, image_size: List[int], out_put_path: str):
        for index, (image, label) in enumerate(self.prepare_unet_images(image_size)):
            pixel = Pixel(10, 10)
            current_label_image = Raster.from_array(label, pixel)
            current_image = Raster.from_array(image, pixel)
            try:
                current_label_image.save_gtiff(out_put_path + f"/label/label_{index}.tif", gdal.GDT_Byte)
                current_image.save_gtiff(out_put_path + f"/image/image_{index}.tif", gdal.GDT_Int16)
            except Exception as e:
                logger.error(e)

    def save_con_images(self):
        pass

    def save_ann_csv(self):
        pass

    def assert_equal_size(self, array1: np.ndarray, array2: np.ndarray):
        shape_1 = ArrayShape(array1.shape)
        shape_2 = ArrayShape(array2.shape)
        try:
            assert shape_1 == shape_2
        except AssertionError:
            raise ValueError("Arrays dont have the same size")


class ImagePlot:

    def __init__(self):
        self._images = []

    def plot(self):
        figure, axes = plt.subplots(nrows=1, ncols=self._images.__len__())
        for index, image in enumerate(self._images):
            try:
                axes[index].imshow(image)
            except TypeError:
                axes.imshow(image)
        plt.show()

    def add(self, array):
        self._images.append(array)


matplotlib.use('Qt5Agg')
if __name__ == "__main__":
    pixel = Pixel(10, 10)
    shape_path = "D:\master_thesis\PYTHON\\test_shape.shp"
    polygon_frame = PolygonFrame.from_file(shape_path)
    union_geom = polygon_frame.union("id")
    #
    point_a = Point(600000, 5611620)
    point_b = Point(662940, 5700000)
    extent = Extent(point_a, point_b)
    label_raster = Raster.from_geo(polygon_frame, extent, pixel)
    main_image = Raster.from_file("D:/master_thesis/data/20160616/img/extent.tif")
    logger.info(f"Current shape is {label_raster.array.shape}")
    img_stand = ImageStand(main_image)
    main_image_stand = img_stand.standarize_image()

    logger.info(main_image_stand.array.shape)
    logger.info(label_raster.array.shape)
    raster_data = RasterData(main_image_stand.array, label_raster.array)
    images = raster_data.prepare_unet_images([500, 500])
    img_plot = ImagePlot()
    img_plot.add(label_raster.array[:, :, 0])
    img_plot.plot()
    # for image, label in images:
    #     if np.unique(label).__len__() > 1:
    #         img_plot = ImagePlot()
    #         logging.info(image.shape)
    #         img_plot.add(image[:, :, [3, 2, 1]])
    #         img_plot.add(label[:, :, 0])
    #         img_plot.plot()


