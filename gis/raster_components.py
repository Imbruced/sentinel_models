import attr
from gis.log_lib import logger
from gis.validators import ispositive
from config.config import Config
from gis.geometry import Point
from gis.geometry import Origin
import numpy as np
import gdal
import os
from osgeo import osr
import ogr
from gis.geometry import GeometryFrame
import matplotlib.pyplot as plt
import json
from typing import List


class StanarizeParams:

    _band_number = attr.ib()
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

    def __init__(self, raster, length, names=None):
        self.__stan_params = StanarizeParams(length)
        self.__names = [f"band_{band}" for band in range(length)] if names is None else names
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
        empty_array = np.empty(shape=[*self.__raster.ref.shape, self.__raster.ref.band_number])

        for index, (band, name) in enumerate(zip(self.__raster.array, self.__names)):
            empty_array[:, :, index] = self.__stand_one_dim(band, name, rescale_function)
            band = None

        ref = ReferencedArray(empty_array, self.__raster.ref.extent, self.__raster.pixel, shape=self.__raster.ref.shape, band_number=self.__raster.ref.band_number)

        return Raster(self.__raster.pixel, ref)

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
        return 7000
        # return array.max()

    @property
    def stan_params(self):
        return self.__stan_params

    @stan_params.setter
    def stan_params(self, value):
        raise AttributeError("This param can' t be set")


class ConfigMeta(type):

    def __new__(mcs, *args, **kwargs):
        x = super().__new__(mcs, *args, **kwargs)
        x.config = Config.from_json(x.__name__.lower())
        return x


@attr.s
class Pixel(metaclass=ConfigMeta):
    x = attr.ib(default=1, validator=[attr.validators.instance_of(float), ispositive])
    y = attr.ib(default=0, validator=[attr.validators.instance_of(float), ispositive])
    unit = attr.ib(default='m', validator=[attr.validators.instance_of(str)])

    @classmethod
    def from_text(cls, text):
        x, y, unit = text.split(" ")
        return cls(int(x), int(y), unit)


@attr.s
class Crs(metaclass=ConfigMeta):
    epsg = attr.ib(default="epsg:4326", validator=[attr.validators.instance_of(str)])


@attr.s
class Extent:

    left_down = attr.ib(default=Point(0, 0))
    right_up = attr.ib(default=Point(1, 1))
    origin = attr.ib(init=False)
    dx = attr.ib(init=False)
    dy = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.origin = Origin(self.left_down.x, self.left_down.y)
        self.dx = self.right_up.dx(self.left_down)
        self.dy = self.right_up.dy(self.left_down)


@attr.s
class ReferencedArray:
    array = attr.ib()
    extent = attr.ib()
    crs = attr.ib()
    shape = attr.ib()
    is_generator = attr.ib(default=False)
    band_number = attr.ib(default=1)


@attr.s
class Raster:

    pixel = attr.ib()
    ref = attr.ib()
    # srs = attr.ib(default=osr.SpatialReference('LOCAL_CS["arbitrary"]'))

    def __attrs_post_init__(self):
        self.array = self.ref.array
        self.extent = self.ref.extent

    def save_gtiff(self, path, raster_dtype):
        drv = gdal.GetDriverByName("GTiff")
        if os.path.isfile(path):
            raise FileExistsError("File currently exists")

        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        band_number = self.array.shape[2]

        ds = drv.Create(path, self.array.shape[1], self.array.shape[0], band_number, raster_dtype)
        transformed_ds = self.__transform(ds, self.extent.origin, self.pixel)
        transformed_ds.SetProjection(WKT_32364)

        for band in range(self.array.shape[2]):
            transformed_ds.GetRasterBand(band + 1).WriteArray(self.array[:, :, band])

    @staticmethod
    def __create_raster(x_shape, y_shape):
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

        raster_ob = cls(pixel, array_copy)
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

        reshaped_array = array.reshape(*array.shape, 1)
        ref = ReferencedArray(array=reshaped_array, crs="2180", extent=extent_new, shape=array.shape[:2])
        raster_ob = cls(pixel=pixel, ref=ref)

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

        array = cls.gdal_file_to_array(raster_from_file)
        band_number = cls.get_band_numbers_gdal(raster_from_file)

        ref = ReferencedArray(array=array, crs="2180", extent=extent, band_number=band_number, shape=[pixel_number_y, pixel_number_x])
        return cls(pixel, ref)

    @staticmethod
    def load_image(path):
        ds: gdal.Dataset = gdal.Open(path)
        return ds

    @staticmethod
    def get_band_numbers_gdal(gdal_image):
        return gdal_image.RasterCount

    @staticmethod
    def gdal_file_to_array(ds):
        for band in range(ds.RasterCount):
            yield ds.GetRasterBand(band+1).ReadAsArray()


class ImagePlot:

    def __init__(self):
        self._images = []

    def plot(self):
        figure, axes = plt.subplots(nrows=int(self._images.__len__()), ncols=2, figsize=(self._images.__len__()*5, 10), sharey =True)
        for index, image in enumerate(self._images):
            logger.info(index)
            axes[index][0].imshow(image[0])
            axes[index][1].imshow(image[1])
        #     try:
        #         axes[index][0].imshow(image[0])
        #         axes[index][1].imshow(image[1])
        #     except TypeError:
        #         axes.imshow(image)
        plt.show()

    def add(self, array):
        self._images.append(array)


@attr.s
class ArrayShape:
    shape = attr.ib()

    def __attrs_post_init__(self):
        self.x_shape = self.shape[0]
        self.y_shape = self.shape[1]

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
        # self.assert_equal_size(array, label_array)
        self.__array = array
        self.__label_array = label_array

    def prepare_unet_images(self, image_size: List[int]):
        x_shape = self.__label_array.shape[1]
        y_shape = self.__label_array.shape[0]
        for curr_x_shape in range(0, y_shape, image_size[0]):
            for curr_y_shape in range(0, x_shape, image_size[1]):
                main_image_sample = self.__array[curr_y_shape: curr_y_shape + image_size[0],
                                                 curr_x_shape: curr_x_shape + image_size[1],
                                                 :]
                label_image_sample = self.__label_array[curr_y_shape: curr_y_shape + image_size[0],
                                                        curr_x_shape: curr_x_shape + image_size[1],
                                                        :]
                if np.unique(label_image_sample).__len__() > 1:

                    yield [main_image_sample, label_image_sample]

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

