import attr
import gdal
from PIL import Image

from exceptions import OptionNotAvailableException
from exceptions.exceptions import FormatNotAvailable
from gis.io_abstract import IoHandler, DefaultOptionWrite
from gis.raster_components import Options
from logs import logger


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
        raster = self.data
        band_number = raster.array.shape[2]
        array = raster.array
        logger.info(self.io_options)
        dtype = self.io_options["dtype"]
        logger.info(dtype)

        ds = drv.Create(path, array.shape[1], array.shape[0], band_number, dtype)
        transformed_ds = raster.transform(ds, raster.extent.origin, raster.pixel)
        for band in range(array.shape[2]):
            transformed_ds.GetRasterBand(band + 1).WriteArray(array[:, :, band])


@attr.s
class PngImageWriter:
    format_name = "png"
    data = attr.ib()
    io_options = attr.ib()

    def save(self, path: str):
        im = Image.fromarray(self.data.array[:, :, 0])
        im.save(path)


