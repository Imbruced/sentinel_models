import attr

from exceptions import OptionNotAvailableException


@attr.s
class Options:
    options = attr.ib(factory=dict)

    def __getitem__(self, item):
        if item in self.options.keys():
            return self.options[item]
        else:
            raise KeyError(f"Can not find {item} in ")

    def __setitem__(self, key, value):
        if key == "format":
            raise AttributeError("format can not be used in options")
        if value is None:
            raise TypeError("Value can not be error")
        if key in self.options.keys():
            self.options[key] = value
        else:
            raise OptionNotAvailableException(f"Can not find option specified in {self.options.keys()}")

    def get(self, item, default=None):
        try:
            value = self.options[item]
            ret_value = value if value is not None else default
        except KeyError:
            raise KeyError(f"Argument {item} is not available")
        return ret_value
