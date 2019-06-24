import re
import os
from typing import List
from typing import NoReturn

import attr
import folium
import geopandas as gpd
import pandas
from shapely import wkt as wkt_loader
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from gis.crs import Crs
from gis.geometry_operations import count_delta
from validators.validators import IsNumeric, WktValidator
from exceptions.exceptions import GeometryCollectionError
from exceptions.exceptions import GeometryTypeError


scriptDirectory = os.path.dirname(os.path.realpath(__file__))


def lazy_property(fn):
    attr_name = '__' + fn.__name__

    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return wrapper


@attr.s
class DataFrameShower:
    MARGIN = 2
    TRUNCATE_SIZE = 20
    data_frame = attr.ib(type=pandas.DataFrame)

    def show(self, limit=5, truncate=True):
        print(self.__create_string(limit, truncate))

    def __create_string(self, limit, truncate):
        columns_length = self.__find_max_length(limit, truncate)
        lines = list()
        lines.append(self.__prepare_row_string(self.data_frame.columns, columns_length))
        limited_data = self.data_frame.head(limit).values.tolist()

        for row in limited_data:
            lines.append(self.__prepare_row_string(row, columns_length))
        dashes = "-" * lines[0].__len__()

        return dashes + "\n" + f"\n{dashes}\n".join(lines) + "\n" + dashes

    @staticmethod
    def __prepare_row_string(row, length_rows):
        rw = []
        for value, length in zip(row, length_rows):
            missing_length = length - len(str(value))
            if missing_length == 0:
                left_add = ""
                right_add = ""
            elif missing_length % 2 == 0:
                left_add = int(missing_length/2) * " "
                right_add = int(missing_length/2) * " "

            else:
                left_add = (int(missing_length / 2) + 1) * " "
                right_add = (int(missing_length / 2)) * " "

            rw.append(DataFrameShower.MARGIN * " " + left_add + f"{str(value)[:length]}" + DataFrameShower.MARGIN * " " + right_add)

        return "|" + "|".join(rw) + "|"

    def __find_max_length(self, limit, truncate):
        maximums = []
        for col in self.data_frame.columns:
            max_length = self.data_frame.head(limit)[col].apply(lambda x: str(x).__len__()).max()
            if truncate:
                maximums.append(min(max_length, self.TRUNCATE_SIZE)+self.MARGIN*2)
            else:
                maximums.append(max_length + self.MARGIN * 2)
        return maximums


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


@attr.s
class InteractiveGeometryPlotter:
    gdf = attr.ib(type=GeometryFrame)

    def plot(self, **kwargs):
        limit = kwargs.get("limit", 100)
        if self.gdf.frame.shape[0] >= 1:
            transformed_frame = self.gdf.transform("epsg:4326")
            first_polygon = transformed_frame.frame["geometry"].head(1).values[0]
            centroid = first_polygon.centroid
            coordinates = [centroid.y, centroid.x]
        else:
            coordinates = [40.7, -74]
        m = folium.Map(coordinates, **kwargs)

        folium.GeoJson(self.gdf.frame.iloc[:limit]).add_to(m)

        return m

    def __prepare_data_for_plotting(self):
        pass


class PointFrame(GeometryFrame):

    def __init__(self, frame, geometry_column):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


class LineFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


class PolygonFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


@attr.s
class Point(ShapelyPoint):
    TYPE = "Point"
    _x = attr.ib(default=0, validator=[IsNumeric()])
    _y = attr.ib(default=0, validator=[IsNumeric()])

    def __attrs_post_init__(self):
        super().__init__(self._x, self._y)

    def __str__(self):
        return f"Point({self.x} {self.y})"

    def translate(self, dx, dy):
        """
        Translates points coordinates by x and y value
        :param x: number of units to move along x axis
        :param y: number of units to move along y axis
        :return: new Point with translated coordinates
        """
        return Point(self.x + dx, self.y + dy)


@attr.s
class PointCollection:
    points = attr.ib(factory=list, type=List[Point])
    crs = attr.ib(default="local")

    def __max_y(self):
        return max([el.y for el in self.points])

    def __max_x(self):
        return max([el.x for el in self.points])

    def __min_x(self):
        return min([el.x for el in self.points])

    def __min_y(self):
        return min([el.y for el in self.points])

    @lazy_property
    def extent(self):
        return Extent(
            Point(self.__min_x(), self.__min_y()),
            Point(self.__max_x(), self.__max_y())
        )


@attr.s
class Polygon(ShapelyPolygon):
    NAME = "Polygon"
    coordinates = attr.ib(type=List[List[float]])
    crs = attr.ib(default=Crs("epsg:4326"))

    def __attrs_post_init__(self):
        super().__init__(self.coordinates)

    @classmethod
    def from_wkt(cls, wkt, crs):
        polygon = wkt_loader.loads(wkt)
        try:
            assert polygon.type == cls.NAME
        except AssertionError:
            raise TypeError(f"wkt is type {polygon.type}")

        return cls(polygon.exterior.coords, crs)


@attr.s
class Line:
    """TODO handle inheritance from Shapely LineString"""
    start = attr.ib()
    end = attr.ib()

    def __attr_post_init__(self):
        self.dx = count_delta(self.start.x, self.start.x)
        self.dy = count_delta(self.start.y, self.end.y)


@attr.s
class Origin(Point):

    _x = attr.ib(default=0)
    _y = attr.ib(default=0)

    def __attr__post_init__(self):
        super().__init__(self._x, self._y)


@attr.s
class Extent:

    left_down = attr.ib(default=Point(0, 0))
    right_up = attr.ib(default=Point(1, 1))
    crs = attr.ib(default=Crs("epsg:4326"))
    origin = attr.ib(init=False)
    dx = attr.ib(init=False)
    dy = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.origin = Origin(self.left_down.x, self.left_down.y)
        self.dx = count_delta(self.left_down.x, self.right_up.x)
        self.dy = count_delta(self.left_down.y, self.right_up.y)

    def transform(self, to_crs):

        pass

    def scale(self, x, y, origin=Point(0, 0)):
        """
        This function takes x and y as the scaling values and divide extent dx and dy by them
        If origin Point is not passed by default it is Point(0, 0)
        :param x: Scaling value x
        :param y: Scaling value y
        :param origin: is the left down corner from which scaled extent will have origin
        :return: returns New instance of extent
        """
        scaled_point = Point(int(self.dx/x + origin.x), int(self.dy/y + origin.y))
        shrinked = Extent(origin, scaled_point)

        return shrinked

    def translate(self, x, y):
        """
        Translates extent coordinates
        :param x:
        :param y:
        :return:
        """

        return Extent(self.left_down, self.right_up.translate(x, y))

    @classmethod
    def from_coordinates(cls, coordinates: List[float], crs="local"):
        point_a = Point(*coordinates[:2])
        point_b = Point(*coordinates[2:])

        return cls(point_a, point_b, crs)

    def expand(self, dx, dy):
        ld = self.left_down.translate(-dx, -dy)
        ru = self.right_up.translate(dx, dy)

        return Extent(ld, ru, crs=self.crs)

    def expand_percentage(self, percent_x, percent_y):
        return self.expand(int(self.dx*percent_x), int(self.dy*percent_y))

    def expand_percentage_equally(self, percent):
        return self.expand_percentage(percent, percent)

    def expand_equally(self, value):
        return self.expand(value, value)

    def to_wkt(self):
        coordinates = [
                        self.left_down,
                        self.left_down.translate(0, self.dy),
                        self.right_up,
                        self.left_down.translate(self.dx, 0),
                        self.left_down
        ]

        coordinates_text = ", ".join([f"{el.x} {el.y}" for el in coordinates])
        return f"POLYGON(({coordinates_text}))"

    def divide_dy(self, tile_size):
        tiles_number_dy = int(float(self.dy) // float(tile_size))
        extents = []
        for tile in range(0, tiles_number_dy):
            extents.append(
                Extent(
                    self.right_up.translate(-self.dx, -(tile+1)*tile_size),
                    self.right_up.translate(0, (-tile) * tile_size),
                    crs=self.crs)
            )
        if int(float(self.dy) // float(tile_size)) != float(self.dy) / float(tile_size):
            extents.append(Extent(
                self.right_up.translate(-self.dx, -self.dy),
                self.right_up.translate(0, -tiles_number_dy*tile_size),
                self.crs
            ))
        return extents

    def divide_dx(self, tile_size):
        tiles_number_dx = int(float(self.dx) // float(tile_size))
        extents = []

        for tile in range(tiles_number_dx):
            extents.append(
                Extent(
                    self.left_down.translate(tile*tile_size, 0),
                    self.left_down.translate((tile+1) * tile_size, self.dy),
                    crs=self.crs)
            )
        if int(float(self.dx) // float(tile_size)) == float(self.dx) / float(tile_size):
            extents.append(Extent(
                self.left_down.translate(tiles_number_dx*tile_size, 0),
                self.left_down.translate(self.dx, self.dy),
                self.crs
            ))

        return extents

    def divide(self, dx, dy):

        if all([dx, dy]):
            dy_divided = self.divide_dy(dy)
            extents = []
            for dy_tile in dy_divided:
                dx_divided = dy_tile.divide_dx(dx)
                for dx_tile in dx_divided:
                    extents.append(dx_tile)
            return extents
        else:
            raise AttributeError("You have to pass all the arguments")
        pass


def cast_float(string: str):
    try:
        casted = float(string)
    except TypeError:
        raise TypeError(f"Can not cast to float value {string}")
    return casted


@attr.s
class Wkt:

    wkt_string = attr.ib(type=str, validator=WktValidator())
    crs = attr.ib(default="local")

    def __attrs_post_init__(self):
        self.__geom = wkt_loader.loads(self.wkt_string)
        self.__type = self.__geom.type
    #
    # def to_geometry(self):
    #     return shapely.wkt.loads(self.wkt_string)

    def __split(self):
        return re.findall(r"(-*\d+\.*\d*) (-*\d+\.*\d*)", self.wkt_string)

    def get_type(self):
        pass

    @lazy_property
    def extent(self) -> Extent:
        return PointCollection(self.coordinates, crs=self.crs).extent

    @lazy_property
    def coordinates(self):
        try:
            coordinates = [Point(cast_float(el[0]), cast_float(el[1])) for el in self.__split()]
        except TypeError:
            raise TypeError("Wrong wkt format")
        except IndexError:
            raise IndexError("Wrong wkt format")
        return coordinates