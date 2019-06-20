
class GeometryCollectionError(Exception):

    def __init__(self, message):
        self.__message = message


class GeometryTypeError(Exception):

    def __init__(self, message):
        self.__message = message


class LessThanZeroException(Exception):

    def __init__(self, message):
        self.message = message


class CrsException(Exception):

    def __init__(self, message):
        super().__init__(message)


class PixelSizeException(Exception):

    def __init__(self, message):
        super().__init__(message)


class ExtensionException(Exception):

    def __init__(self, message):
        super().__init__(message)


class ConfigException(Exception):

    def __init__(self, message):
        super().__init__(message)


class OptionNotAvailableException(Exception):
    def __init__(self, message):
        super().__init__(message)


class FormatNotAvailable(Exception):
    def __init__(self, message):
        super().__init__(message)


class DimensionException(Exception):
    def __init__(self, message):
        super().__init__(message)
