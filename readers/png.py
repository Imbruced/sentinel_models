import attr

from readers.reader import Reader


@attr.s
class PngImageReader(Reader):
    format_name = "png"