import attr

from config.options import Options


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