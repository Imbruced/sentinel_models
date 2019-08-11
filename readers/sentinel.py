import attr



@attr.s
class SentinelImageReader:
    io_options = attr.ib()
    format_name = "sentinel"
    data = attr.ib()

    def __attrs_post_init__(self):
        extent_wkt = self.io_options["extent"].wkt
        user = User(self.io_options["user"], self.io_options["password"])
        self.sentinel_downloader = SentinelDownloader(
            user,
            extent_wkt,
            self.io_options["date"]
        )

    def load(self):
        pass

    def download_files(self):
        self.sentinel_downloader.download_items()

    def _wrap_files(self):
        pass

    def _merge_files(self):
        GdalMerge(
            arguments=[
                "-ps", "10", "10", "-separate"
            ],
            files=[
            ],
        )

    def __get_request_metadata(self):
        return self.sentinel_downloader.scenes