import attr
import folium

import matplotlib.pyplot as plt


@attr.s
class SubPlots:
    images = attr.ib(factory=list)

    def plot(self, nrows):
        image_numbers = self.images.__len__()
        try:
            ncols = image_numbers // nrows
        except ZeroDivisionError:
            raise ZeroDivisionError("Value for nrows must be greater than 0")
        if ncols == 0:
            raise ValueError("Invalid dimension")
        figure, axes = plt.subplots(nrows=nrows,
                                    ncols=ncols,
                                    figsize=(self.images.__len__() * 5, 10))

        for index, image in enumerate(self.images):
            curr_column = (index + ncols) % ncols
            curr_row = index // ncols
            if ncols == 1 and nrows > 1:
                current_axis = axes[curr_row]
            elif nrows == 1 and ncols > 1:
                current_axis = axes[curr_column]
            elif nrows > 1 and ncols > 1:
                current_axis = axes[curr_row][curr_column]
            else:
                raise ValueError("Invalid Dimension")

            try:
                if image.shape[2] == 3:
                    current_axis.imshow(image.array[:, :, :3])
                elif image.shape[2] == 1:
                    current_axis.imshow(image.array[:, :, 0])
            except IndexError:
                raise ValueError("Invalid dimension for data")

        plt.show()

    def add(self, raster):
        return self.__class__([*self.images, raster])

    def extend(self, *rasters):
        return self.__class__([*self.images, *rasters])


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


@attr.s
class InteractiveGeometryPlotter:
    gdf = attr.ib(type='GeometryFrame')

    def plot(self, **kwargs):
        limit = kwargs.get("limit", 100)
        if self.gdf.frame.shape[0] >= 1:
            transformed_frame = self.gdf.transform("epsg:4326")
            first_polygon = transformed_frame.frame["geometry"].head(1).values[0]
            centroid = first_polygon.centroid
            coordinates = [centroid.y, centroid.x]
        else:
            coordinates = [40.7, -74]
        m = folium.Map(coordinates, **kwargs)

        folium.GeoJson(self.gdf.frame.iloc[:limit]).add_to(m)

        return m

    def __prepare_data_for_plotting(self):
        pass