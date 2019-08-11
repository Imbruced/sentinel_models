import attr
from PIL import Image

from writers.writer import Writer


@attr.s
class PngImageWriter(Writer):
    format_name = "png"
    data = attr.ib()
    io_options = attr.ib()

    def save(self, path: str):
        im = Image.fromarray(self.data[:, :, 0])
        im.save(path)
