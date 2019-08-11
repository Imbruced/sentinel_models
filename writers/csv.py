from typing import NoReturn

import attr

from writers.writer import Writer


@attr.s
class CsvImageWriter(Writer):
    io_options = attr.ib()
    format_name = "csv"
    data = attr.ib()

    def save(self, path: str) -> NoReturn:
        """TODO add most common option to csv writer"""
        label_data = self.__add_label_data_if_its_provided()
        pass

    def __add_label_data_if_its_provided(self):
        from gis.raster import Raster
        label_data = self.io_options["label_data"]
        if not isinstance(label_data, Raster):
            raise TypeError("label data has to be raster type")
        return label_data