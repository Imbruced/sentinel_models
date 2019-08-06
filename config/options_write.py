import attr
import gdal

from config.options import Options


@attr.s
class DefaultOptionWrite:

    @classmethod
    def csv(cls):
        return Options(
            {
                "format": "csv",
                "delimiter": ",",
                "header": ","
            }
        )

    @classmethod
    def geotiff(cls):
        return Options(
            {
                "format": "geotiff",
                "dtype": gdal.GDT_Byte,
                "crs": None
            }
        )

    @classmethod
    def png(cls):
        return Options(
            {
                "format": "png"
            }
        )

    @classmethod
    def shapefile(cls):
        pass

    @classmethod
    def wkt(cls):
        pass