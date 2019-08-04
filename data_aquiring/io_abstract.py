import inspect
import sys

import attr
import gdal

from exceptions import OptionNotAvailableException


@attr.s
class Options:
    options = attr.ib(factory=dict)

    def __getitem__(self, item):
        if item in self.options.keys():
            return self.options[item]
        else:
            raise KeyError(f"Can not find {item} in ")

    def __setitem__(self, key, value):
        if key == "format":
            raise AttributeError("format can not be used in options")
        if value is None:
            raise TypeError("Value can not be error")
        if key in self.options.keys():
            self.options[key] = value
        else:
            raise OptionNotAvailableException(f"Can not find option specified in {self.options.keys()}")

    def __eq__(self, other):
        return self.options == other.options

    def get(self, item, default=None):
        try:
            value = self.options[item]
            ret_value = value if value is not None else default
        except KeyError:
            raise KeyError(f"Argument {item} is not available")
        return ret_value


@attr.s
class ClsFinder:

    name = attr.ib(type="str")

    def __get_cls_tuples(self):
        return inspect.getmembers(sys.modules[self.name], inspect.isclass)

    @property
    def available_cls(self):
        __cls = []
        for cls in self.__get_cls_tuples():
            try:
                __cls.append(cls[1])
            except IndexError:
                pass
        return __cls


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


@attr.s
class DefaultOptionRead:

    @classmethod
    def wkt(cls):
        return Options(
            {
                "format": "wkt",
                "crs": None,
                "pixel": None,
                "value": 1,
                "extent": None
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
    def geotiff(cls):
        return Options(
            {
                "format": "geotiff",
                "crs": None
            }
        )

    @classmethod
    def shp(cls):
        return Options(
            {
                "format": "shp",
                "type": None,
                "driver": "ESRI Shapefile",
                "crs": None,
                "extent": None,
                "pixel": None,
                "value": 1,
                "all_unique": "False",
                "color_column": None
            }
        )

    @classmethod
    def postgis_geom(cls):
        return Options(
            {
                "format": "postgis_geom",
                "type": None,
                "crs": None,
                "extent": None,
                "pixel": None,
                "value": 1,
                "all_unique": "False",
                "color_column": None,
                "schema": "public",
                "host": "localhost",
                "user": "postgres",
                "password": "postgres"
            }
        )

    @classmethod
    def geojson(cls):
        return Options(
            {
                "format": "shp",
                "driver": "GeoJSON",
                "type": None,
                "crs": None,
                "extent": None,
                "pixel": None,
                "value": 1,
                "all_unique": "False",
                "color_column": None
            }
        )

    @classmethod
    def sentinel(cls):
        return Options(
            {
                "format": "sentinel",
                "extent": None,
                "user": None,
                "password": None,
                "pixel": None
            }
        )


