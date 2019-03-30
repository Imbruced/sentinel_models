from gis.log_lib import logger
import matplotlib.pyplot as plt


class ImagePlot:

    def __init__(self):
        self._images = []

    def plot(self):
        figure, axes = plt.subplots(nrows=int(self._images.__len__()),
                                    ncols=2,
                                    figsize=(self._images.__len__()*5, 10),
                                    sharey="true")

        for index, image in enumerate(self._images):
            logger.info(index)
            axes[index][0].imshow(image[0])
            axes[index][1].imshow(image[1])
        plt.show()

    def add(self, array):
        self._images.append(array)
