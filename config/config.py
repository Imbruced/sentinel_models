import attr
import json
import os

script_path = os.path.split(os.path.realpath(__file__))[0]


@attr.s
class Config:
    config = attr.ib(factory=list)

    @classmethod
    def from_json(cls, name):
        return cls(cls.__load_json(name))

    @staticmethod
    def __load_json(name):
        with open(os.path.join(script_path, name+".json")) as f:
            data = json.load(f)
        return data

