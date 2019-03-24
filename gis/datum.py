class Crs:

    def __init__(self, epsg):
        self.__epsg = epsg

    @classmethod
    def from_proj4(cls):
        pass

    @classmethod
    def from_epsg(self):
        pass

    @property
    def epsg(self):
        return self.__epsg