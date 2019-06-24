import attr

from gis.meta import ConfigMeta


@attr.s
class Crs(metaclass=ConfigMeta):
    epsg = attr.ib(default="epsg:4326", validator=[attr.validators.instance_of(str)])

    @classmethod
    def from_proj4(cls):
        pass

    @classmethod
    def from_epsg(self):
        pass

    def __str__(self):
        return self.epsg

    @property
    def code(self):
        return int(self.epsg.split(":")[1])


def load_crs_list():
    crs_list = []
    return crs_list


CRS = load_crs_list()

