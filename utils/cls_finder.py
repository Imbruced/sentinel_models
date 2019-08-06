import inspect
import sys
from typing import List, Any

import attr


@attr.s
class ClsFinder:

    name = attr.ib(type=str)

    def __get_cls_tuples(self) -> List[Any]:
        return inspect.getmembers(sys.modules[self.name], inspect.isclass)

    @property
    def available_cls(self) -> List[Any]:
        __cls = []
        for cls in self.__get_cls_tuples():
            try:
                __cls.append(cls[1])
            except IndexError:
                pass
        return __cls