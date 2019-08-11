from readers.from_geometry import ShapeImageReader
from readers.from_geometry import PostgisGeomImageReader
from readers.from_geometry import GeoJsonImageReader
from readers.geotiff import GeoTiffImageReader
from readers.from_geometry import WktImageReader
# from readers.sentinel import SentinelImageReader

readers = [
    ShapeImageReader,
    PostgisGeomImageReader,
    GeoTiffImageReader,
    GeoTiffImageReader,
    WktImageReader
]

__all__ = ['readers']