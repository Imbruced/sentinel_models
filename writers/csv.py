from typing import NoReturn

import attr
import pandas as pd

from gis.raster import Raster
from preprocessing.data_preparation import AnnDataCreator


@attr.s
class CsvImageWriter:
    io_options = attr.ib()
    format_name = "csv"
    data = attr.ib()

    def save(self, path: str) -> NoReturn:
        """TODO add most common option to csv writer"""
        label_data = self.__add_label_data_if_its_provided()
        ann_data = AnnDataCreator(
            image=self.data,
            label=label_data
        ).concat_arrays()

        pd.DataFrame(ann_data).\
            to_csv(path, sep=self.io_options["delimiter"], index=False)

    def __add_label_data_if_its_provided(self):
        label_data = self.io_options["label_data"]
        if not isinstance(label_data, Raster):
            raise TypeError("label data has to be raster type")
        return label_data