from abc import ABC
from copy import deepcopy
from typing import Dict, List
import abc

import attr


@attr.s
class IoHandler(ABC):
    io_options = attr.ib()

    def options(self, **kwargs):
        current_options = deepcopy(self.io_options)
        for key in kwargs:
            current_options[key] = kwargs[key]
        return current_options

    @abc.abstractmethod
    def available_cls(self, regex: str, name: str) -> List['IoHandler']:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cls(self, regex: str, name: str) -> Dict[str, 'IoHandler']:
        raise NotImplementedError
