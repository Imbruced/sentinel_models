import shutil
from datetime import datetime
from unittest import TestCase

from gis.preprocessing import SentinelDownloader, User, unzip_sentinel_data_based_on_key, move_the_data_and_remove, \
    move_the_data_full_process_based_on_key, GdalMerge

USER = "pkocinski001"
PASSWORD = "Kojack.0116!"
wkt = "POLYGON((18.601393 53.022649, 18.600420 53.025014, 18.604536 53.025043, 18.604523 53.022647, 18.601393 53.022649))"


class TestSentinelApi(TestCase):
    date = datetime(2019, 4, 6, 10, 0, 29, 24000)
    sentinel_downloader = SentinelDownloader(User(USER, PASSWORD), wkt, date)

    def test_get_items_metadata(self):
        self.assertTrue(type(self.sentinel_downloader.items_metadata), dict())

    def test_get_closest_date(self):
        print(self.sentinel_downloader.scenes)

    def test_download_closest_date(self):
        self.sentinel_downloader.download_items()

    def test_unzip_data(self):
        unzip_sentinel_data_based_on_key("50a91dc1-b140-45fd-9cdf-9cd66b920ac4", r".*_B\d{1}\w{1}_\d{2}m.*")

    def test_move_files(self):
        try:
            shutil.rmtree("C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\sentinel_out\\50a91dc1-b140-45fd-9cdf-9cd66b920ac")
        except FileNotFoundError:
            pass
        unzip_sentinel_data_based_on_key("50a91dc1-b140-45fd-9cdf-9cd66b920ac4", r".*_B\d{1}\w{1}_\d{2}m.*")
        move_the_data_and_remove("50a91dc1-b140-45fd-9cdf-9cd66b920ac4")

    def test_move_files_if_they_exists(self):
        with self.assertRaises(TypeError):
            unzip_sentinel_data_based_on_key("50a91dc1-b140-45fd-9cdf-9cd66b920ac4", r".*_B\d{1}\w{1}_\d{2}m.*")
            try:
                move_the_data_and_remove("50a91dc1-b140-45fd-9cdf-9cd66b920ac4")
            except TypeError:
                shutil.rmtree(
                "C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\sentinel_out\\50a91dc1-b140-45fd-9cdf-9cd66b920ac4_copy")
                raise TypeError("Not empty directory")

    def test_full_pipeline(self):
        with self.assertRaises(TypeError):
            move_the_data_full_process_based_on_key("50a91dc1-b140-45fd-9cdf-9cd66b920ac4")

    def test_merge_image(self):
        GdalMerge(
            arguments=[
                "-ps", "10", "10", "-separate"
            ],
            files=[
                "C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\sentinel_out\\50a91dc1-b140-45fd-9cdf-9cd66b920ac4\\R10m\\T33UYU_20190406T100029_B02_10m.jp2",
                "C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\sentinel_out\\50a91dc1-b140-45fd-9cdf-9cd66b920ac4\\R10m\\T33UYU_20190406T100029_B03_10m.jp2",
                "C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\sentinel_out\\50a91dc1-b140-45fd-9cdf-9cd66b920ac4\\R10m\\T33UYU_20190406T100029_B04_10m.jp2",
                "C:\\Users\\Pawel\\Desktop\\sentinel_models\\data\\sentinel_out\\50a91dc1-b140-45fd-9cdf-9cd66b920ac4\\R10m\\T33UYU_20190406T100029_B08_10m.jp2"
            ],
            output_location="C:\\Users\\Pawel\\Desktop\\sentinel_models\data\\sentinel_out\\res.tiff"
        ).execute()