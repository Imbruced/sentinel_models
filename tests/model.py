import os

import matplotlib
from gis.geometry import PolygonFrame
from gis.raster import Raster
from gis.standarizer import ImageStand
from sklearn.preprocessing import StandardScaler
from gis.data_prep import RasterData

from models import Unet
from models import UnetConfig
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from plotting import ImagePlot

matplotlib.use('Qt5Agg')

if __name__ == "__main__":

    module_path = os.path.split(os.getcwd())[0]

    shape_path = os.path.join(module_path, "data", "domy.shp")
    image_path = os.path.join(module_path, "data", "buildings.tif")
    compiled_model = os.path.join(module_path, "data", "model_more_class_pixels.h5")

    polygon_frame = PolygonFrame.from_file(shape_path).union("id")

    main_image = Raster.from_file(image_path, crs="epsg:26917")
    label_raster = Raster.with_adjustment("from_geo", main_image, polygon_frame)

    standarize1 = ImageStand(raster=main_image)
    raster_data = RasterData(standarize1.standarize_image(StandardScaler()), label_raster)
    unet_images = raster_data.prepare_unet_images(image_size=(64, 64))
    callbacks = [
        EarlyStopping(patience=100, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=100, min_lr=0, verbose=1),
        ModelCheckpoint('model_more_class_pixels.h5', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    config = UnetConfig(
        input_size=[64, 64, 3],
        callbacks=callbacks
    )

    unet = Unet(config=config)
    unet = Unet.load_from_weight_file(compiled_model, config=config)
    predicted = unet.predict(x=unet_images.x_test[:10, :, :, :], threshold=0.6)

    image_plot = ImagePlot()
    image_plot.extend(
        [
            unet_images.x_test[1, :, :, :3],
            unet_images.y_test[1, :, :, 0],
            predicted[1, :, :, 0],
            unet_images.x_test[2, :, :, :3],
            unet_images.y_test[2, :, :, 0],
            predicted[2, :, :, 0],
            unet_images.x_test[3, :, :, :3],
            unet_images.y_test[3, :, :, 0],
            predicted[3, :, :, 0],
            unet_images.x_test[4, :, :, :3],
            unet_images.y_test[4, :, :, 0],
            predicted[4, :, :, 0],
            unet_images.x_test[5, :, :, :3],
            unet_images.y_test[5, :, :, 0],
            predicted[5, :, :, 0]
        ]
    )
    image_plot.plot(nrows=5)