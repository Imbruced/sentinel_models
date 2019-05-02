from gis.raster_components import Pixel
import attr
import os
from gis.geometry import Point
from gis.geometry import Origin
from osgeo import osr
import ogr
from gis.geometry import GeometryFrame
from gis.geometry import Extent
from typing import Union
from copy import copy
import gdal
from gis.raster_components import ReferencedArray
from gis.log_lib import logger
from gis.exceptions import CrsException
from logs.log_lib import logger
from gis.image_loader import GdalImage


@attr.s
class Raster:
    pixel = attr.ib()
    ref = attr.ib()

    # srs = attr.ib(default=osr.SpatialReference('LOCAL_CS["arbitrary"]'))

    def __attrs_post_init__(self):
        self.array = self.ref.array
        self.extent = self.ref.extent

    def save_png(self, path):

        scipy.misc.imsave(path + 'outfile.jpg', self.array)

    def save_gtiff(self, path, raster_dtype):
        """
        TODO delete hardcoded values and use existing classes to simplify the code
        This method is not production ready it needs to be simplified and properly
        rewritten
        :param path:
        :param raster_dtype:
        :return:
        """
        drv = gdal.GetDriverByName("GTiff")
        # path = Path(path)
        # if not path.is_file():
        #     raise FileNotFoundError()
        #
        # if os.path.isfile(path):
        #     raise FileExistsError("File currently exists")
        logger.info(path)
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0], exist_ok=True)

        band_number = self.array.shape[2]

        ds = drv.Create(path, self.array.shape[1], self.array.shape[0], band_number, raster_dtype)
        transformed_ds = self.__transform(ds, self.extent.origin, self.pixel)
        # transformed_ds.SetProjection()

        for band in range(self.array.shape[2]):
            logger.info("Writing an array")
            transformed_ds.GetRasterBand(band + 1).WriteArray(self.array[:, :, band])

    @classmethod
    def empty_raster(cls, extent, pixel):
        transformed_raster, extent_new = cls.__gdal_raster_from_extent(extent, pixel)
        return cls.__convert_gdal_raster_raster_instance(transformed_raster, extent_new, pixel)

    @classmethod
    def from_wkt(cls, geometry: str, extent, pixel):
        """
        TODO simplify this method, remove hardcoded crs
        TODO add crs validation, write class representing crsError
        This method converts passed as string wkt into raster format to the extent
        Remember to pass wkt in the same coordinate reference system as in extent
        :param geometry:
        :param extent:
        :param pixel:
        :return:
        """
        transformed_raster, extent_new = cls.__gdal_raster_from_extent(extent, pixel)
        transformed_raster = cls.__insert_polygon(transformed_raster, geometry, 1)
        return cls.__convert_gdal_raster_raster_instance(transformed_raster, extent_new, pixel)

    @classmethod
    def with_adjustment(cls, method: str, existing_raster, geometry: Union[GeometryFrame, str]):
        """
        Function allows to use existing raster metadata such as pixel size and extent arguments
        to adjust geometry into it
        :param method: Currently supported methods from_wkt and from geo
        :param existing_raster: it instance of class Raster
        :param geometry: is a wkt or GeometryFrame instance which will be converted into raster
        :return: new Raster with geometry adjusted into existing raster metadata
        """

        pixel = copy(existing_raster.pixel)
        extent = existing_raster.ref.extent

        return getattr(cls, method)(geometry, extent, pixel)

    @classmethod
    def from_geo(cls, geometry: GeometryFrame, extent: Extent, pixel: Pixel):

        """
        TODO simplify this method and looka at extent new object
        :param geoframe:
        :param extent:
        :param pixel:
        :return:
        """
        wkt_frame = geometry.to_wkt()
        wkt_strings = wkt_frame.frame["wkt"].values.tolist()

        if not wkt_strings:
            logger.warning("Provided an empty geodataframe, raster will be created from and extent only")
            return cls.empty_raster(extent, pixel)

        transformed_raster, extent_new = cls.__gdal_raster_from_extent(extent, pixel)
        logger.info(f"Geometry CRS: {geometry.crs}")
        logger.info(f"Extent CRS: {extent.crs}")
        if geometry.crs != extent.crs:
            logger.error(f"incompatible crs between extent and geometry "
                         f"frame geometry crs {geometry.crs} and extent crs: {extent.crs}")
            raise CrsException("Extent crs is not the same as geometry frame crs, please give the same ")

        for index, wkt_string in enumerate(wkt_strings):
            cls.__insert_polygon(transformed_raster, wkt_string, index+1)

        return cls.__convert_gdal_raster_raster_instance(transformed_raster, extent_new, pixel)

    @classmethod
    def from_file(cls, path, crs="local"):
        """
        Based on provided path to raster instance of class Raster will be created
        :param path: str path to raster file, look at possible extension in GdalImage class
        :param crs: Coordinate reference system
        :return: Instance of Raster
        """
        gdal_image = GdalImage.load_from_file(path, crs)
        logger.info(gdal_image.array.shape)
        ref = ReferencedArray(array=gdal_image.array.transpose([1, 2, 0]),
                              crs=crs, extent=gdal_image.extent,
                              band_number=gdal_image.band_number,
                              shape=[gdal_image.pixel_size_y, gdal_image.pixel_size_x])

        return cls(pixel=gdal_image.pixel, ref=ref)

    @staticmethod
    def __create_raster(x_shape, y_shape):
        memory_ob = gdal.GetDriverByName('MEM')
        raster = memory_ob.Create('', x_shape, y_shape, 1, gdal.GDT_Byte)

        return raster

    @staticmethod
    def __transform(raster, origin: Origin, pixel: Pixel):
        copy_raster = raster
        copy_raster.SetGeoTransform((origin.x, pixel.x, 0.0, origin.y + (raster.RasterYSize * pixel.y), 0, -pixel.y))
        left_top_corner_x, pixel_size_x, _, left_top_corner_y, _, pixel_size_y = copy_raster.GetGeoTransform()
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
    def __gdal_raster_from_extent(cls, extent, pixel):
        """
        This method based on extent instance and pixel value creating empty raster

        :param extent: instance of Extent
        :param pixel: instance of pixel
        :return: gdal raster prepared based on specified extent
        """

        new_extent = extent.scale(pixel.x, pixel.y)

        extent_new = Extent(Point(extent.origin.x, extent.origin.y),
                            Point((new_extent.origin.x + new_extent.dx) * pixel.x,
                                  (new_extent.origin.y + new_extent.dy) * pixel.y))

        raster = cls.__create_raster(new_extent.dx, new_extent.dy)
        transformed_raster = cls.__transform(raster, extent.origin, pixel)

        return transformed_raster, extent_new

    @classmethod
    def __convert_gdal_raster_raster_instance(cls, transformed_raster, extent, pixel):
        array = transformed_raster.ReadAsArray()
        reshaped_array = array.reshape(*array.shape, 1)
        ref = ReferencedArray(array=reshaped_array, crs="2180", extent=extent, shape=array.shape[:2])
        raster_ob = cls(pixel=pixel, ref=ref)

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
        ref = ReferencedArray(array=array_copy, crs=extent.crs, extent=extent, shape=array.shape[:2])
        raster_ob = cls(pixel, ref)
        return raster_ob


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

    def prepare_unet_images(self, image_size: List[int], remove_empty_labels=True):
        chunks_array = create_two_dim_chunk(self.image.array, image_size)
        chunks_label = create_two_dim_chunk(self.label.array, image_size)

        for img, lbl in zip(chunks_array, chunks_label):
            if remove_empty_labels or np.unique(lbl).__len__() > 1:
                yield [img, lbl]

    def save_unet_images(self, image_size: List[int], out_put_path: str):
        for index, (image, label) in enumerate(self.prepare_unet_images(image_size)):
            logger.info(image.shape)
            current_label_image = Raster.from_array(label, self.image.pixel)
            current_image = Raster.from_array(image, self.label.pixel)
            try:
                current_label_image.save_gtiff(out_put_path + f"/label/label_{index}.tif", gdal.GDT_Byte)
                current_image.save_gtiff(out_put_path + f"/image/image_{index}.tif", gdal.GDT_Int16)
            except Exception as e:
                logger.error(e)

    @classmethod
    def from_path(cls, path: str, label_name: str, image_name: str):
        label_path = os.path.join(path, label_name)
        image_path = os.path.join(path, image_name)
        pass

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
