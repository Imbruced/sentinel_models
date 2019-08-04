import attr

from validators.validators import IsImageFile


@attr.s
class ImageFile:

    """
    Class prepared for image path validation
    """

    path = attr.ib(validator=[IsImageFile()])

    def __str__(self):
        return self.path