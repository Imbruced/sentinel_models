from abc import ABC

from preprocessing.image_standarizer import ImageStand


class Standarizer(ABC):
    stand = ImageStand
