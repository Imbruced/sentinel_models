from abc import ABC

import attr

from exceptions import CrsException
from gis import Pixel, Extent, Point, GeometryFrame, Raster
from gis.gdal_image import GdalImage
from gis.geometry import Wkt


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
