import unittest
from gis.raster import Pixel


class TestPixel(unittest.TestCase):

    def test_pixel(self):
        pixel = Pixel(10, 10)
        self.assertEqual(10, 10)

    def test_negative_pixel(self):
        with self.assertRaises(ValueError):
            pixel = Pixel(-10, -10)

    def test_neg_pos_pixel(self):
        with self.assertRaises(ValueError):
            pixel = Pixel(-10, 10)

    def test_null_pixel(self):
        with self.assertRaises(ValueError):
            pixel1 = Pixel(None, None)


if __name__ == '__main__':
    unittest.main()
