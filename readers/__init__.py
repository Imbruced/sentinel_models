from readers.from_geometry import ShapeImageReader
from readers.from_geometry import PostgisGeomImageReader
from readers.from_geometry import GeoJsonImageReader
from readers.geotiff import GeoTiffImageReader
# from readers.sentinel import SentinelImageReader

readers = [
    ShapeImageReader,
    PostgisGeomImageReader,
    GeoTiffImageReader,
    GeoTiffImageReader
]

__all__ = ['readers']