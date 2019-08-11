from typing import Tuple

import attr
import gdal
import numpy as np
import osr
import ogr

from exceptions.exceptions import DimensionException
from gis import Point, Extent
from gis.crs import Crs
from gis.raster_components import ReferencedArray
from preprocessing.abstract import ImageFile


@attr.s
class GdalImage:

    ds = attr.ib(type=gdal.Dataset)
    path = attr.ib(default=None)
    crs = attr.ib(default=Crs("local"))

    def __attrs_post_init__(self):
        from gis import Pixel
        self.__transform_params = self.ds.GetGeoTransform()
        self.left_x = self.__transform_params[0]
        self.pixel_size_x = self.__transform_params[1]
        self.top_y = self.__transform_params[3]
        self.pixel_size_y = -self.__transform_params[5]
        self.x_size = self.ds.RasterXSize
        self.y_size = self.ds.RasterYSize
        self.pixel = Pixel(abs(self.pixel_size_x), abs(self.pixel_size_y))
        self.extent = Extent.from_coordinates([
            self.left_x, self.top_y - (self.y_size * abs(self.pixel_size_y)),
            self.left_x + abs(self.pixel_size_x) * self.x_size, self.top_y
        ], self.crs)
        self.band_number = self.ds.RasterCount

    @classmethod
    def load_from_file(cls, path: str, crs=None):
        """
        class method which based on path returns GdalImage object,
        It validates path location and its format
        :return: GdalImage instance
        """
        file = ImageFile(path)
        ds: gdal.Dataset = gdal.Open(file.path)
        if crs is None:
            projection = ds.GetProjection()
            srs = osr.SpatialReference(wkt=projection)
            crs_gdal = srs.GetAttrValue('AUTHORITY', 0).lower() + ":" + srs.GetAttrValue('AUTHORITY', 1)
            crs = c.Crs(crs_gdal)
        return cls(ds, path, crs)

    @classmethod
    def in_memory(cls, x_shape, y_shape, crs=Crs("epsg:4326")):
        memory_ob = gdal.GetDriverByName('MEM')
        raster = memory_ob.Create('', x_shape, y_shape, 1, gdal.GDT_Byte)
        if raster is None:
            raise ValueError("Your image is to huge, please increase pixel size, by using pixel option in loading options, example pixel=Pixel(20.0, 20.0)")
        return cls(raster, crs=crs)

    @classmethod
    def from_extent(cls, extent, pixel):
        new_extent = extent.scale(pixel.x, pixel.y)

        extent_new = Extent(Point(extent.origin.x, extent.origin.y),
                            Point((new_extent.origin.x + new_extent.dx) * pixel.x,
                                  (new_extent.origin.y + new_extent.dy) * pixel.y))

        raster = cls.in_memory(int(new_extent.dx), int(new_extent.dy), crs=extent.crs)

        transformed_raster = raster.transform(extent.origin, pixel)

        return transformed_raster, extent_new

    def __read_as_array(self, ds: gdal.Dataset) -> np.ndarray:
        if not hasattr(self, "__array"):
            setattr(self, "__array", ds.ReadAsArray())
        return getattr(self, "__array")

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    @property
    def array(self) -> np.ndarray:
        """
        Property which returns numpy.ndarray representation of file
        :return: np.ndarray
        """
        return self.__read_as_array(self.ds)

    def to_raster(self) -> Tuple['Pixel', 'ReferencedArray']:
        if self.array.shape.__len__() == 3:
            arr = self.array
        elif self.array.shape.__len__() == 2:
            arr = self.array.reshape(1, *self.array.shape)
        else:
            raise DimensionException("Array should be shape 2 or 3")

        ref = ReferencedArray(
            array=arr.transpose([1, 2, 0]),
            crs=self.crs,
            extent=self.extent,
            band_number=self.band_number,
            shape=[self.pixel_size_y,
                   self.pixel_size_x]
        )
        return self.pixel, ref

    def transform(self, origin: 'Origin', pixel: 'Pixel', projection='LOCAL_CS["arbitrary"]'):
        self.ds.SetGeoTransform((origin.x, pixel.x, 0.0, origin.y + (self.y_size * pixel.y), 0, -pixel.y))
        left_top_corner_x, pixel_size_x, _, left_top_corner_y, _, pixel_size_y = self.ds.GetGeoTransform()
        self.ds.SetProjection(projection)

        return GdalImage(self.ds, crs=self.crs)

    def insert_polygon(self, wkt, value):
        srs = osr.SpatialReference('LOCAL_CS["arbitrary"]')
        rast_ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('wrk')
        rast_mem_lyr = rast_ogr_ds.CreateLayer('poly', srs=srs)
        feat = ogr.Feature(rast_mem_lyr.GetLayerDefn())
        feat.SetGeometryDirectly(ogr.Geometry(wkt=wkt))
        rast_mem_lyr.CreateFeature(feat)
        err = gdal.RasterizeLayer(self.ds, [1], rast_mem_lyr, None, None, [value], ['ALL_TOUCHED=TRUE'])

        return GdalImage(self.ds)