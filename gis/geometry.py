import os
from typing import NoReturn

import attr
import geopandas as gpd

from gis.crs import Crs
from exceptions.exceptions import GeometryCollectionError
from exceptions.exceptions import GeometryTypeError
from plotting.frame_shower import DataFrameShower
from plotting.plotting import InteractiveGeometryPlotter

scriptDirectory = os.path.dirname(os.path.realpath(__file__))


@attr.s
class GeometryFrame:
    frame = attr.ib()
    geometry_column = attr.ib()
    crs = attr.ib(default=Crs("epsg:4326"))

    def __attr__post_init__(self):
        self.type = self.__class__.__name__

    def to_wkt(self) -> 'GeometryFrame':
        """
        Based on geometry column this method returns wkt representation of it, it does not modify
        passed instance, it create new one

        :return: GeometryFrame with new wkt column
        """

        frame_copy = self.frame.copy()
        frame_copy["wkt"] = frame_copy["geometry"].apply(lambda x: x.wkt)

        return self.__class__(frame_copy, "geometry", crs=self.crs)

    @classmethod
    def from_file(cls, path, driver="ESRI Shapefile") -> 'GeometryFrame':
        """
        :param path: File location
        :param driver: Driver for operning the file, currentyl supported drivers are ex.
        ESRI Shapefile
        GeoJSON
        GPKG
        OpenFileGDB
        For all  supported drivers look at
        ```python
            import fiona
            fiona.supported_drivers
        ```
        :return: GeometryFrame
        """
        geometry = gpd.read_file(path, driver=driver)
        GeoFrame = cls(geometry, "geom", Crs(geometry.crs["init"]))

        return GeoFrame

    def transform(self, crs) -> 'GeometryFrame':
        """
        Transform GeometryFrame to other coordinate reference system
        :param crs: crs code example 'epsg:2180'
        :return: GeometryFrame with target coordinate reference system
        """
        return self.__class__(
            self.frame.to_crs({"init": crs}),
            crs=crs,
            geometry_column=self.geometry_column
        )

    def union(self, attribute) -> 'GeometryFrame':
        """

        :param attribute: name of column based on which dissolve will be done on geometry column
        :return: GeometryFrame
        """
        dissolved = self.frame.dissolve(by=attribute, aggfunc='sum')
        geoframe = self.__class__(dissolved, "geometry")
        geoframe.type = "Multi" + self.type
        geoframe.crs = self.crs
        return geoframe

    def _assert_geom_type(self):
        unique_geometries = [el for el in set(self.frame.type) if el is not None]
        if unique_geometries.__len__() != 1:
            raise GeometryCollectionError("Object can not be collection of geometries")

        try:
            assert str(list(set(self.frame.type))[0]) + "Frame" == self.type
        except AssertionError:
            raise GeometryTypeError("Your input geometry type is incorrect")

    def show(self, limit=5, truncate=True) -> NoReturn:
        """
        This function will show GeometryFrame in Apache Spark ASCII style
        :param limit: how many rows to show
        :param truncate: decide if record values should be truncated to 20 chars
        :return: NoReturn
        """
        DataFrameShower(self.frame).show(limit, truncate)

    def plot(self, interactive=False, **kwargs) -> NoReturn:
        """

        :param interactive: decide if this function should use folium or geopandas plotting function
        if Interactive is set to True than folium library will be used if False than geopandas plotting
        library.
        :param kwargs: Other arguments for plotting, look at geoframe plot function and folium Map constructor
        :return: NoReturn
        """
        if interactive:
            return InteractiveGeometryPlotter(self).plot(**kwargs)
        else:
            self.frame.plot(**kwargs)

    def head(self, limit=5):
        """
        This function returns number of rows specified in argument limit
        :param limit: How many rows to return
        :return:
        """
        return self.frame.head(limit)


def cast_float(string: str):
    try:
        casted = float(string)
    except TypeError:
        raise TypeError(f"Can not cast to float value {string}")
    return casted
