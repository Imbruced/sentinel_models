import json
import os
import re
from abc import ABC
from copy import deepcopy
from typing import Tuple, NoReturn, Dict

import numpy as np
from PIL import Image
import attr
import gdal
import osr
import pandas as pd

from data_aquiring.downloading import GdalMerge, SentinelDownloader, User
from data_aquiring import io_abstract as ia
from data_aquiring.io_abstract import DefaultOptionWrite, Options, ClsFinder
from exceptions.exceptions import FormatNotAvailable, OptionNotAvailableException, CrsException
from gis.geometry import Extent, Point
from gis.crs import Crs
from gis.decorators import classproperty
from gis.gdal_image import GdalImage
from gis.geometry import lazy_property, GeometryFrame, Wkt
from gis.raster_components import ReferencedArray, Pixel
from plotting import ImagePlot
from preprocessing import data_preparation as dp
from preprocessing.scalers import RangeScaler, StanarizeParams


@attr.s
class IoHandler(ABC):
    io_options = attr.ib(type=Options)

    def options(self, **kwargs):
        current_options = deepcopy(self.io_options)
        for key in kwargs:
            current_options[key] = kwargs[key]
        return current_options

    def available_cls(self, regex: str, name: str):
        if not hasattr(self, "__writers"):
            setattr(self, "__writers", self.__get_cls(regex, name))
        return getattr(self, "__writers")

    def __get_cls(self, regex: str, name: str) -> Dict[str, str]:
        return {cl.format_name: cl
                for cl in ClsFinder(name).available_cls
                if re.match(regex, cl.__name__)
                }


@attr.s
class CsvImageWriter:
    io_options = attr.ib()
    format_name = "csv"
    data = attr.ib()

    def save(self, path: str) -> NoReturn:
        """TODO add most common option to csv writer"""
        label_data = self.__add_label_data_if_its_provided()
        ann_data = dp.AnnDataCreator(
            image=self.data,
            label=label_data
        ).concat_arrays()

        pd.DataFrame(ann_data).\
            to_csv(path, delimiter=self.io_options["delimiter"], index=False)

    def __add_label_data_if_its_provided(self):
        label_data = self.io_options["label_data"]
        if not isinstance(label_data, Raster):
            raise TypeError("label data has to be raster type")
        return label_data


@attr.s
class SentinelImageReader:
    io_options = attr.ib()
    format_name = "sentinel"
    data = attr.ib()

    def __attrs_post_init__(self):
        extent_wkt = self.io_options["extent"].wkt
        user = User(self.io_options["user"], self.io_options["password"])
        self.sentinel_downloader = SentinelDownloader(
            user,
            extent_wkt,
            self.io_options["date"]
        )

    def load(self):
        pass

    def download_files(self):
        self.sentinel_downloader.download_items()

    def _wrap_files(self):
        pass

    def _merge_files(self):
        GdalMerge(
            arguments=[
                "-ps", "10", "10", "-separate"
            ],
            files=[
            ],
        )

    def __get_request_metadata(self):
        return self.sentinel_downloader.scenes


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
            default_options = getattr(ia.DefaultOptionWrite, format)()
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
        gdal_raster = GdalImage(ds, self.data.crs)
        srs = osr.SpatialReference()  # Establish its coordinate encoding
        srs.ImportFromEPSG(self.data.crs.code)
        transformed_ds = gdal_raster.transform(self.data.extent.origin, self.data.pixel, srs.ExportToWkt())

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
            default_options = getattr(ia.DefaultOptionRead, format)()
        except AttributeError:
            raise FormatNotAvailable("Can not found requested format")
        return self.__class__(
            io_options=default_options
        )


@attr.s
class ImageReader(Reader):

    io_options = attr.ib(type=Options, default=getattr(ia.DefaultOptionRead, "wkt")())

    def load(self, path) -> 'Raster':
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

        pixel, ref = gdal_image.to_raster()

        return Raster(pixel=pixel, ref=ref)


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

    def load(self):
        geoframe = GeometryFrame.from_file(self.path, self.io_options["driver"]).to_wkt()

        crs = self.io_options.get("crs", geoframe.crs)

        gdf = self.__add_value_column(geoframe)

        wkt_value_list = [[Wkt(el[0]), el[1]] for el in gdf[["wkt", "raster_value"]].values.tolist()]

        extent = self._find_extent_from_multiple_wkt(wkt_value_list, crs=crs)

        gdal_raster = self.wkt_to_gdal_raster(extent, self.io_options)

        for wkt, value in wkt_value_list:
            gdal_raster.insert_polygon(wkt.wkt_string, value)
        pixel, ref = gdal_raster.to_raster()
        return Raster(pixel=pixel, ref=ref)

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
class ShapeImageReader(RasterFromGeometryReader):
    format_name = "shp"


@attr.s
class GeoJsonImageReader(RasterFromGeometryReader):
    format_name = "geojson"


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

        pixel, ref = gdal_raster.to_raster()
        return Raster(pixel=pixel, ref=ref)


@attr.s
class PostgisGeomImageReader(RasterFromGeometryReader):
    format_name = "postgis_geom"

    def load(self):
        pass


@attr.s
class RasterCreator:

    @classmethod
    def empty_raster(cls, extent: Extent, pixel: Pixel) -> Tuple[Pixel, ReferencedArray]:
        transformed_raster, extent_new = GdalImage.from_extent(extent, pixel)
        return cls.to_raster(transformed_raster, pixel)

    @staticmethod
    def to_raster(gdal_raster, pixel) -> Tuple[Pixel, ReferencedArray]:
        array = gdal_raster.ds.ReadAsArray()
        reshaped_array = array.reshape(*array.shape, 1)
        ref = ReferencedArray(array=reshaped_array,
                              crs=gdal_raster.crs,
                              extent=gdal_raster.extent,
                              shape=array.shape[:2])
        return pixel, ref


class Raster(np.ndarray):
    """
    This class directly inherits from numpy array, so all functionality from numpy array is available,
    In addition this class preserve Geospatial information. It can be written to geotiff format with
    geo reference. After predicting raster can be saved with georeference.
    It allows to load data from image data types such as GeoTiff and PNG, but also it have possibility
    to load the data from geometry fromats like ShapeFile, GeoJSon, Wkt and convert them to raster format
    with preserving geospatial information. It's important that easly it can be aligned into existing raster.
    It is important when creating label and image dataset for andy CNN based model.
    """

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
    def read(self) -> 'ImageReader':
        """
        This function returns ImageReader which allows you to read the data from various formats in
        Apache Spark style, Example:
        Raster\
            .read\
            .format("shp") \
            .options(pixel=Pixel(1.0, 1.0))
            .load("file_location")
        The shape file will be loaded from path specified with pixel 1.0 1.0 in size
        To align to exsiting raster you can write the code in below example
        image = Raster\
                    .read\
                    .format("geotiff")\
                    .load("image_path")
        label = Raster\
                    .read\
                    .format("shp")\
                    .options(
                            pixel=image.pixel,
                            extent=image.extent
                    )
        This will create label image based on shapefile with the same number of pixels. It's important
        that your geometry has the same crs with image, if not Exception will be raised. Currently transformation
        on the fly is not supported.
        :return: 'ImageReader'
        """
        return ImageReader()

    @property
    def write(self) -> 'ImageWriter':
        """
        This function allows to write data into formats like:
        PNG
        GEoTiff
        based on raster data
        example
        image\
            .write\
            .format("geotiff")\
            .save("file_path")
        Reference will be created for GeoTiff file based on extent and crs instance attributes
        :return: 'ImageWriter'
        """
        return ImageWriter(data=self)

    @classmethod
    def from_array(cls, array, pixel, extent=Extent(Point(0, 0), Point(1, 1))):
        array_copy = array
        ref = ReferencedArray(array=array_copy, crs=extent.crs, extent=extent, shape=array.shape[:2])
        raster_ob = cls(pixel, ref)
        return raster_ob

    @classmethod
    def empty(cls, extent: Extent = Extent(), pixel: Pixel = Pixel(0.1, 0.1)):
        raster_ob = RasterCreator.empty_raster(pixel=pixel, extent=extent)
        return Raster(pixel=raster_ob[0], ref=raster_ob[1])

    @property
    def array(self):
        return self.ref.array

    def show(self, true_color=False):
        print(self)
        if not true_color:
            plotter = ImagePlot(self[:, :, 0])
        else:
            array = np.array(
                [
                    ImageStand(self[:, :, 0:1]).standarize_image(RangeScaler()),
                    ImageStand(self[:, :, 1:2]).standarize_image(RangeScaler()),
                    ImageStand(self[:, :, 2:3]).standarize_image(RangeScaler())
                ]
            ).transpose([1, 2, 0, 3]).reshape(*self.shape)
            plotter = ImagePlot(array)
        plotter.plot()

    @lazy_property
    def extent(self):
        return self.ref.extent

    @lazy_property
    def crs(self):
        return self.extent.crs


@attr.s
class ImageStand:
    """
    Class is responsible for scaling image data, it requires gis.raster.Raster object
    Band names can be passed, by default is ["band_index" ... "band_index+n"]
    use standarize_image method to standarize data, it will return Raster with proper refactored image values

    """
    stan_params = attr.ib(init=False)
    raster = attr.ib()
    names = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.stan_params = StanarizeParams(self.raster.ref.band_number)
        self.names = [f"band_{band}" for band in range(self.raster.shape[-1])] \
            if self.names is None else self.names

    def save_params(self, path):
        if os.path.exists(path):
            raise FileExistsError("File with this name exists in directory")
        with open(path, "w") as file:
            params_json = json.load(self.stan_params.coefficients)
            file.writelines(params_json)

    def standarize_image(self, scaler=None):
        """
        Rescale data by maximum
        :return:
        """
        empty_array = np.empty(shape=[*self.raster.shape])

        for index, name in enumerate(self.names):
            empty_array[:, :, index] = self.__stand_one_dim(self.raster[:, :, index], name, scaler)
            band = None

        ref = ReferencedArray(empty_array,
                              self.raster.ref.extent,
                              self.raster.pixel,
                              shape=self.raster.ref.shape,
                              band_number=self.raster.ref.band_number)

        return Raster(self.raster.pixel, ref)

    def __stand_one_dim(self, array: np.ndarray, band_name: str, scaler):
        """

        :param array:
        :param band_name:
        :param scaler:
        :return:
        """

        fitted = scaler.fit(array)

        self.stan_params.coefficients[band_name] = fitted

        self.stan_params.add(fitted, band_name)
        return fitted.transform(array)