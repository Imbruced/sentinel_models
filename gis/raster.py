import os
from copy import copy
from typing import Union

import scipy.misc
import attr
from osgeo import osr
import ogr
import gdal

from gis.geometry import Point
from gis.geometry import Origin
from gis.geometry import GeometryFrame
from gis.geometry import Extent
from gis.raster_components import ReferencedArray
from gis.raster_components import Pixel
from exceptions.exceptions import CrsException
from gis.writers import ImageWriter
from logs.log_lib import logger
from gis.image_loader import GdalImage


@attr.s
class Raster:
    pixel = attr.ib()
    ref = attr.ib()

    def __attrs_post_init__(self):
        self.array = self.ref.array
        self.extent = self.ref.extent

    def write(self):
        """

        :param path:
        :param raster_dtype:
        :return:
        """

        return ImageWriter(data=self)

    @classmethod
    def empty_raster(cls, extent: Extent = Extent(), pixel: Pixel = Pixel(0.1, 0.1)):
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
    def transform(raster, origin: Origin, pixel: Pixel):
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
        transformed_raster = cls.transform(raster, extent.origin, pixel)

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



