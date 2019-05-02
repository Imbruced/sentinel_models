from validators import UnetImageShape
from unittest import TestCase


class TestValidators(TestCase):

    def test_unet_shape_list(self):
        shape = [128, 128, 3]
        test_shape = UnetImageShape()(value=shape)

    def test_unet_shape_tuple(self):
        shape = (128, 128, 3)
        test_shape = UnetImageShape()(value=shape)

    def test_unet_shape_not_tuple_and_list(self):
        shape = None
        with self.assertRaises(TypeError):
            test_shape = UnetImageShape()(value=shape)

    def test_unet_shape_len_less_than_three(self):
        with self.assertRaises(AttributeError):
            shape = (128, 128)
            test_shape1 = UnetImageShape()(value=shape)

            shape1 = (128, 128, 4, 7)
            test_shape2 = UnetImageShape()(value=shape1)

            shape2 = ()
            test_shape3 = UnetImageShape()(value=shape2)

    def test_unet_shape_non_integer(self):

        with self.assertRaises(TypeError):
            shape = (128, 128, "string")
            test_shape1 = UnetImageShape()(value=shape)

            shape1 = (128, 128, None)
            test_shape2 = UnetImageShape()(value=shape1)

            shape2 = ("128", "128", "tuple")
            test_shape3 = UnetImageShape()(value=shape2)

            shape3 = (128.0, 128.0, 4.0)
            test_shape4 = UnetImageShape()(value=shape3)

    def test_last_dimension(self):
        with self.assertRaises(AttributeError):
            shape = (128, 128, 0)
            test_shape1 = UnetImageShape()(value=shape)

    def test_bad_image_shapes(self):
        with self.assertRaises(ValueError):
            shape = (128, 63, 10)
            test_shape1 = UnetImageShape()(value=shape)

            shape = (128, 63, 10)
            test_shape1 = UnetImageShape()(value=shape)

            shape = (63, 128, 10)
            test_shape1 = UnetImageShape()(value=shape)

            shape = (63, 63, 10)
            test_shape1 = UnetImageShape()(value=shape)

            shape = (0, 0, 10)
            test_shape1 = UnetImageShape()(value=shape)



