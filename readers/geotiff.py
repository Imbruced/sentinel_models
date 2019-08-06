import attr

from gis.gdal_image import GdalImage
from gis.raster import Raster


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