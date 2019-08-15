from abc import ABC
from typing import List

import attr
import numpy as np
from sklearn.model_selection import train_test_split

from gis import Raster


@attr.s
class ModelData(ABC):
    x = attr.ib(type=List[Raster])
    y = attr.ib(type=List[Raster])
    test_size = attr.ib(default=0.15)
    random_state = attr.ib(default=2018)

    def __attrs_post_init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.number_of_classes = np.unique(self.y)


@attr.s
class UnetData(ModelData):
    test_size = attr.ib(default=0.1)
    random_state = attr.ib(default=2018)


class AnnData(ModelData):

    test_size = attr.ib(default=0.1)
    random_state = attr.ib(default=2018)


class CnnData(ModelData):

    test_size = attr.ib(default=0.1)
    random_state = attr.ib(default=2018)
