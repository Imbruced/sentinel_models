from typing import Tuple, NoReturn

import numpy as np
import attr

from gis.crs import Crs
from gis.extent import Extent
from gis.point import Point
from interfaces.image_stand import Standarizer
from readers.image import ImageReaderFactory
from utils.decorators import classproperty, lazy_property
from gis.gdal_image import GdalImage
from gis.raster_components import ReferencedArray, Pixel
from plotting import ImagePlot
from preprocessing.scalers import RangeScaler
from writers.image import ImageWriterFactory


@attr.s
class RasterCreator:

    @classmethod
    def empty_raster(cls, extent: 'Extent', pixel: Pixel) -> Tuple[Pixel, ReferencedArray]:
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
    def read(self) -> ImageReaderFactory:
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
        return ImageReaderFactory()

    @property
    def write(self) -> ImageWriterFactory:
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
        :return: 'ImageWriterFactory'
        """
        return ImageWriterFactory(data=self)

    @classmethod
    def from_array(cls, array, pixel, extent=Extent(Point(0, 0), Point(1, 1))) -> 'Raster':
        array_copy = array
        ref = ReferencedArray(array=array_copy, crs=extent.crs, extent=extent, shape=array.shape[:2])
        return Raster(pixel, ref)

    @classmethod
    def empty(cls, extent: Extent = Extent(), pixel: Pixel = Pixel(0.1, 0.1)) -> 'Raster':
        raster_ob = RasterCreator.empty_raster(pixel=pixel, extent=extent)
        return Raster(pixel=raster_ob[0], ref=raster_ob[1])

    @property
    def array(self):
        return self.ref.array

    def show(self, true_color: bool = False) -> NoReturn:
        if not true_color:
            plotter = ImagePlot(self[:, :, 0])
        else:
            array = np.array(
                [
                    Standarizer.stand(self[:, :, 0:1]).standarize_image(RangeScaler()),
                    Standarizer.stand(self[:, :, 1:2]).standarize_image(RangeScaler()),
                    Standarizer.stand(self[:, :, 2:3]).standarize_image(RangeScaler())
                ]
            ).transpose([1, 2, 0, 3]).reshape(*self.shape)
            plotter = ImagePlot(array)
        plotter.plot()

    @lazy_property
    def extent(self) -> Extent:
        return self.ref.extent

    @lazy_property
    def crs(self) -> Crs:
        return self.extent.crs
