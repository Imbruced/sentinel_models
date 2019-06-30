import inspect
import re
import sys
from abc import ABC
from copy import deepcopy
from typing import Dict

import attr
import gdal

from gis.raster_components import Options


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
                "password": None
            }
        )


