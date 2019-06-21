from typing import Union

import attr

from logs.log_lib import logger
import matplotlib.pyplot as plt

@attr.s
class SubPlots:
    images = attr.ib()

    def plot(self, nrows):
        image_numbers = self.images.__len__()
        ncols = image_numbers // nrows

        figure, axes = plt.subplots(nrows=nrows,
                                    ncols=ncols,
                                    figsize=(self.images.__len__() * 5, 10),
                                    sharey="all")

        for index, image in enumerate(self.images):
            curr_column = (index + ncols) % ncols
            curr_row = index // ncols
            axes[curr_row][curr_column].imshow(image)
        plt.show()

    def add(self, array):
        return self.__class__([*self.images, array])

    def extend(self, *arrays):
        return self.__class__([*self.images, *arrays])


@attr.s
class ImagePlot:

    image = attr.ib()

    def plot(self):
        figure = plt.figure()
        axis = figure.add_axes([0, 0, 1, 1])
        axis.imshow(self.image)
        plt.show()

    def add(self, array):
        return SubPlots([self.image, array])

    def extend(self, *arrays):
        return SubPlots([self.image, *arrays])
