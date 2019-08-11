from writers.csv import CsvImageWriter
from writers.geo_tiff import GeoTiffImageWriter
from writers.png import PngImageWriter

writers = [
    CsvImageWriter,
    GeoTiffImageWriter,
    PngImageWriter
]

__all__ = ['writers']

