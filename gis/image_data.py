import re
from abc import ABC

import ogr
import osr
from PIL import Image
import attr
import gdal
import numpy as np

from gis.crs import Crs
from gis.decorators import classproperty
from gis.io_abstract import IoHandler, DefaultOptionWrite
from gis.geometry import Extent, Point, Origin, Wkt, GeometryFrame, lazy_property
from gis.raster_components import Pixel
from plotting import ImagePlot
from validators.validators import IsImageFile
from exceptions.exceptions import FormatNotAvailable, OptionNotAvailableException, DimensionException, CrsException
from gis.io_abstract import DefaultOptionRead
from gis.raster_components import Options, ReferencedArray


@attr.s
class ImageFile:

    """
    Class prepared for image path validation
    """

    path = attr.ib(validator=[IsImageFile()])

    def __str__(self):
        return self.path


@attr.s
class GdalImage:

    ds = attr.ib(type=gdal.Dataset)
    path = attr.ib(default=None)
    crs = attr.ib(default="local")

    def __attrs_post_init__(self):
        self.__transform_params = self.ds.GetGeoTransform()
        self.left_x = self.__transform_params[0]
        self.pixel_size_x = self.__transform_params[1]
        self.top_y = self.__transform_params[3]
        self.pixel_size_y = -self.__transform_params[5]
        self.x_size = self.ds.RasterXSize
        self.y_size = self.ds.RasterYSize
        self.pixel = Pixel(abs(self.pixel_size_x), abs(self.pixel_size_y))
        self.extent = Extent.from_coordinates([
            self.left_x, self.top_y - (self.y_size * self.pixel_size_y),
            self.left_x + self.pixel_size_x * self.x_size, self.top_y
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
            crs = srs.GetAttrValue('AUTHORITY', 0).lower() + ":" + srs.GetAttrValue('AUTHORITY', 1)
        return cls(ds, path, Crs(crs))

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

    def to_raster(self):
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
        return Raster(pixel=self.pixel, ref=ref)

    def transform(self, origin: Origin, pixel: Pixel):
        self.ds.SetGeoTransform((origin.x, pixel.x, 0.0, origin.y + (self.y_size * pixel.y), 0, -pixel.y))
        left_top_corner_x, pixel_size_x, _, left_top_corner_y, _, pixel_size_y = self.ds.GetGeoTransform()
        self.ds.SetProjection('LOCAL_CS["arbitrary"]')

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


@attr.s
class RasterCreator:

    def empty_raster(self, extent: Extent, pixel: Pixel):
        transformed_raster, extent_new = GdalImage.from_extent(extent, pixel)
        return self.to_raster(transformed_raster, pixel)

    @staticmethod
    def to_raster(gdal_raster, pixel):
        array = gdal_raster.ds.ReadAsArray()
        reshaped_array = array.reshape(*array.shape, 1)
        ref = ReferencedArray(array=reshaped_array,
                              crs=gdal_raster.crs,
                              extent=gdal_raster.extent,
                              shape=array.shape[:2])
        raster_ob = Raster(pixel=pixel, ref=ref)

        return raster_ob


class Raster(np.ndarray):
    raster_creator = RasterCreator()

    def __new__(cls, pixel, ref):
        obj = np.asarray(ref.array).view(cls)
        obj.pixel = pixel
        obj.ref = ref
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.pixel = getattr(obj, 'pixel', None)
        self.ref = getattr(obj, 'ref', None)

    def __array_wrap__(self, out_arr, context=None):
        return super().__array_wrap__(self, out_arr, context)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, self.__class__):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, self.__class__):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super().__array_ufunc__(ufunc, method,
                                                 *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], self.__class__):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(self.__class__)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], self.__class__):
            results[0].info = info

        return results[0] if len(results) == 1 else results

    @classproperty
    def read(self):
        return ImageReader()

    @property
    def write(self):
        return ImageWriter(data=self)

    @classmethod
    def from_array(cls, array, pixel, extent=Extent(Point(0, 0), Point(1, 1))):
        array_copy = array
        ref = ReferencedArray(array=array_copy, crs=extent.crs, extent=extent, shape=array.shape[:2])
        raster_ob = cls(pixel, ref)
        return raster_ob

    @classmethod
    def empty(cls, extent: Extent = Extent(), pixel: Pixel = Pixel(0.1, 0.1)):
        return cls.raster_creator.empty_raster(extent, pixel)

    @property
    def array(self):
        return self.ref.array

    def show(self, true_color=False):
        if not true_color:
            plotter = ImagePlot(self[:, :, 0])
        else:
            plotter = ImagePlot(self[:, :, :3])
        plotter.plot()

    @lazy_property
    def extent(self):
        return self.ref.extent

    @lazy_property
    def crs(self):
        return self.extent.crs


@attr.s
class Writer(IoHandler):
    data = attr.ib()
    io_options = attr.ib(type=Options())

    def save(self, path: str):
        raise NotImplemented()

    def options(self, **kwargs):
        current_options = super().options(**kwargs)
        return self.__class__(data=self.data, io_options=current_options)

    def format(self, format):
        try:
            default_options = getattr(DefaultOptionWrite, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")

        return self.__class__(
            data=self.data,
            io_options=default_options
        )


@attr.s
class ImageWriter(Writer):
    data = attr.ib()
    io_options = attr.ib(default=getattr(DefaultOptionWrite, "geotiff")())

    def save(self, path: str):
        image_format = self.__get_writer()
        writer = self.writers[image_format](
            io_options=self.io_options,
            data=self.data
        )
        writer.save(path)

    def __get_writer(self):
        image_format = self.io_options["format"]
        if image_format not in self.writers:
            raise OptionNotAvailableException(f"Option {image_format} is not implemented \n available options {self.__str_writers}")
        return image_format

    @property
    def writers(self):
        return self.available_cls(r"(\w+)ImageWriter", __name__)

    @property
    def __str_writers(self):
        return ", ".join(self.available_cls(r"(\w+)ImageWriter", __name__))


@attr.s
class GeoTiffImageWriter:
    format_name = "geotiff"
    data = attr.ib()
    io_options = attr.ib()

    def save(self, path: str):
        drv = gdal.GetDriverByName("GTiff")
        band_number = self.data.shape[2]
        dtype = self.io_options["dtype"]

        ds = drv.Create(path, self.data.shape[1], self.data.shape[0], band_number, dtype)
        gdal_raster = GdalImage(ds, "local")
        transformed_ds = gdal_raster.transform(gdal_raster.extent.origin, gdal_raster.pixel)
        for band in range(self.data.shape[2]):
            transformed_ds.ds.GetRasterBand(band + 1).WriteArray(self.data[:, :, band])


@attr.s
class PngImageWriter:
    format_name = "png"
    data = attr.ib()
    io_options = attr.ib()

    def save(self, path: str):
        im = Image.fromarray(self.data[:, :, 0])
        im.save(path)


@attr.s
class Reader(IoHandler):
    io_options = attr.ib(type=Options)

    def load(self, path: str):
        return NotImplemented()

    def options(self, **kwargs):
        options = super().options(**kwargs)
        return self.__class__(options)

    def format(self, format):
        try:
            default_options = getattr(DefaultOptionRead, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")
        return self.__class__(
            io_options=default_options
        )


@attr.s
class ImageReader(Reader):

    io_options = attr.ib(type=Options, default=getattr(DefaultOptionRead, "wkt")())

    def load(self, path) -> Raster:
        image_format = self.__get_reader()
        reader = self.readers[image_format](
            io_options=self.io_options,
            path=path
        ).load()

        return reader

    @property
    def readers(self):
        return self.available_cls(r"(\w+)ImageReader", __name__)

    def __str_readers(self):
        return ", ".join(self.available_cls(r"(\w+)ImageReader", __name__))

    def __get_reader(self):
        """TODO to simplify or move to upper class"""
        image_format = self.io_options["format"]
        if image_format not in self.readers:
            raise OptionNotAvailableException(f"Option {image_format} is not implemented \n available options {self.__str_readers}")
        return image_format


@attr.s
class GeoTiffImageReader:

    path = attr.ib()
    io_options = attr.ib()
    format_name = "geotiff"

    def load(self):

        gdal_image = GdalImage.load_from_file(
            self.path,
            self.io_options["crs"])

        return gdal_image.to_raster()


@attr.s
class PngImageReader:
    format_name = "png"


@attr.s
class RasterFromGeometryReader(ABC):
    path = attr.ib()
    io_options = attr.ib()

    @classmethod
    def wkt_to_gdal_raster(cls, default_extent, options):
        extent = options.get(
            "extent",
            default_extent.expand_percentage_equally(0.3)
        )
        try:
            assert extent.crs == default_extent.crs
        except AssertionError:
            raise CrsException("Crs from extent does not match with Crs specified. Please make changes.")

        pixel: Pixel = options.get("pixel", Pixel(0.5, 0.5))
        gdal_in_memory, extent_new = GdalImage.from_extent(
            extent, pixel
        )

        return gdal_in_memory

    @classmethod
    def _find_extent_from_multiple_wkt(cls, wkt_value_list, crs=Crs("epsg:4326")):
        bottom_corners = [el[0].extent.left_down for el in wkt_value_list]
        top_corners = [el[0].extent.right_up for el in wkt_value_list]

        min_x = min(bottom_corners, key=lambda x: x.x).x
        min_y = min(bottom_corners, key=lambda x: x.y).y
        max_x = max(top_corners, key=lambda x: x.x).x
        max_y = max(top_corners, key=lambda x: x.y).y

        extent = Extent(
            Point(min_x, min_y),
            Point(max_x, max_y),
            crs=crs
        )
        return extent


@attr.s
class ShapeImageReader(RasterFromGeometryReader):
    format_name = "shp"

    def load(self):
        geoframe = GeometryFrame.from_file(self.path).to_wkt()

        crs = self.io_options.get("crs", geoframe.crs)

        gdf = self.__add_value_column(geoframe)

        wkt_value_list = [[Wkt(el[0]), el[1]] for el in gdf[["wkt", "raster_value"]].values.tolist()]

        extent = self._find_extent_from_multiple_wkt(wkt_value_list, crs=crs)

        gdal_raster = self.wkt_to_gdal_raster(extent, self.io_options)

        for wkt, value in wkt_value_list:
            gdal_raster.insert_polygon(wkt.wkt_string, value)
        return gdal_raster.to_raster()

    def __add_value_column(self, gdf: GeometryFrame):
        all_unique = self.io_options["all_unique"]
        gdf = gdf.frame
        if self.io_options["color_column"] is not None:
            gdf["raster_value"] = gdf[self.io_options["color_column"]]
        elif all_unique == "True":
            try:
                gdf.drop_column(["index"], axis=1)
            except AttributeError:
                pass
            gdf = gdf.reset_index()
            gdf = gdf.rename(columns={"index": "raster_value"})
        elif all_unique == "False":
            gdf["raster_value"] = self.io_options["value"]

        return gdf


@attr.s
class WktImageReader(RasterFromGeometryReader):
    format_name = "wkt"

    def load(self):
        wkt: Wkt = Wkt(self.path)
        gdal_raster = self.wkt_to_gdal_raster(wkt.extent, self.io_options)

        gdal_raster.insert_polygon(
            wkt.wkt_string,
            self.io_options["value"]
        )
        return gdal_raster.to_raster()


@attr.s
class PostgisGeomImageReader(RasterFromGeometryReader):
    format_name = "postgis_geom"

    def load(self):
        pass
