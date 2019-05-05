from typing import Union

from logs.log_lib import logger
import matplotlib.pyplot as plt


class ImagePlot:

    def __init__(self):
        self._images = []

    def plot(self, nrows=1):
        image_numbers = self._images.__len__()
        ncols = image_numbers // nrows
        logger.info(f"Cols number: {ncols}")
        logger.info(f"Row number {nrows}")

        figure, axes = plt.subplots(nrows=nrows,
                                    ncols=ncols,
                                    figsize=(self._images.__len__()*5, 10),
                                    sharey="all")

        for index, image in enumerate(self._images):
            logger.info(index)
            curr_column = (index + ncols) % ncols
            curr_row = index // ncols
            axes[curr_row][curr_column].imshow(image)
        plt.show()

    def add(self, array):
        self._images.append(array)

    def extend(self, images: Union[list, tuple]):
        self._images.extend(list(images))
