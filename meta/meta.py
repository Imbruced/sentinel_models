from config.config import Config


class ConfigMeta(type):

    def __new__(mcs, *args, **kwargs):
        x = super().__new__(mcs, *args, **kwargs)
        x.config = Config.from_json(x.__name__.lower())
        return x
