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
from gis.raster_components import Path
from gis.exceptions import CrsException


@attr.s
class Raster:
    pixel = attr.ib()
    ref = attr.ib()

    # srs = attr.ib(default=osr.SpatialReference('LOCAL_CS["arbitrary"]'))

    def __attrs_post_init__(self):
        self.array = self.ref.array
        self.extent = self.ref.extent

    def __save_gtiff(self, path, raster_dtype):
        """
        TODO delete hardcoded values and use existing classes to simplify the code
        This method is not production ready it needs to be simplified and properly
        rewritten
        :param path:
        :param raster_dtype:
        :return:
        """
        drv = gdal.GetDriverByName("GTiff")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError()

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

        if geometry.crs != extent.crs:
            logger.error("incompatible crs between extent and geometry frame")
            raise CrsException("Extent crs is not the same as geometry frame crs, please give the same ")

        for index, wkt_string in enumerate(wkt_strings):
            cls.__insert_polygon(transformed_raster, wkt_string, index)

        return cls.__convert_gdal_raster_raster_instance(transformed_raster, extent_new, pixel)

    @classmethod
    def from_file(cls, path):

        """
        TODO This method needs simplifying
        :param path:
        :return:
        """
        raster_from_file = cls.load_image(path)
        left_top_corner_x, pixel_size_x, _, left_top_corner_y, _, pixel_size_y = raster_from_file.GetGeoTransform()

        pixel_number_x = raster_from_file.RasterXSize
        pixel_number_y = raster_from_file.RasterYSize

        pixel = Pixel(pixel_size_x, -pixel_size_y)

        extent = Extent(Point(left_top_corner_x, left_top_corner_y - -(pixel_number_y * pixel_size_y)),
                        Point(left_top_corner_x + pixel_number_x * pixel_size_x, left_top_corner_y))

        array = cls.gdal_file_to_array(raster_from_file)
        band_number = cls.get_band_numbers_gdal(raster_from_file)

        ref = ReferencedArray(array=array, crs="2180", extent=extent, band_number=band_number,
                              shape=[pixel_number_y, pixel_number_x])

        return cls(pixel, ref)

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

        raster_ob = cls(pixel, array_copy)
        return raster_ob

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
            yield ds.GetRasterBand(band + 1).ReadAsArray()
