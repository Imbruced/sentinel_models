from unittest import TestCase
from models.unet import Unet
from models.unet import UnetConfig


class UnetTest(TestCase):

    def test_default_init(self):
        unet = Unet()
        unet.compile()
        unet.summary()

    def test_init(self):
        unet_config = UnetConfig(
            input_size=(16, 16, 3),
            filters=10,
            dropout=0.6,
            batchnorm=False
        )
        unet = Unet(config=unet_config)
        unet.compile(loss="binary_crossentropy", metrics=["accuracy"])
        unet.summary()


