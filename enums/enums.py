from enum import Enum, unique


@unique
class Metrics(Enum):
    pass


@unique
class Loss(Enum):
    pass


@unique
class ConfDictAttributes(Enum):
    at = "activations"
    la = "layers"
    ind = "input_dim"