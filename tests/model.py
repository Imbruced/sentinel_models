import matplotlib
from gis.geometry import PolygonFrame
from gis.raster_components import Pixel
from gis.geometry import Point
from gis.raster_components import Extent
from gis.raster_components import Raster
from gis.standarizer import ImageStand
from gis.plotting import ImagePlot
from gis.log_lib import logger
from gis.raster_components import RasterData
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc


matplotlib.use('Qt5Agg')
if __name__ == "__main__":
    pixel = Pixel(10.0, 10.0)
    shape_path = "D:\\master_thesis\\data\\20160616\\geom\\class_5.shp"
    polygon_frame = PolygonFrame.from_file(shape_path)
    polygon_frame = polygon_frame.union("new_class")
    point_a = Point(600000, 5611620)
    point_b = Point(662940, 5700000)
    extent = Extent(point_a, point_b)
    label_raster = Raster.from_geo(polygon_frame, extent, pixel)
    main_image = Raster.from_file("D:/master_thesis/data/20160616/img/extent.tif")
    img_stand = ImageStand(main_image, 13)
    main_image_stand = img_stand.standarize_image()

    logger.info("Summary Shapes")
    logger.info(main_image_stand.array.shape)
    logger.info(label_raster.array.shape)
    raster_data = RasterData(main_image_stand.array, label_raster.array)
    images = raster_data.prepare_unet_images([128, 128])

    labels = np.array([el[1] for el in images])
    images = raster_data.prepare_unet_images([128, 128])
    images_main = np.array([el[0] for el in images])

    for index, el in enumerate(images_main):
        logger.info(el.shape)

    logger.info(labels.shape)
    logger.info(images_main.shape)
    img_stand = None
    main_image_stand = None
    label_raster = None
    main_image = None
    raster_data = None
    images = None

    logger.info("Releasing memory")
    import time
    time.sleep(5)

    #keras part
    from keras.layers import Input
    from models.unet import get_unet
    from keras.optimizers import Adam
    from models.metrics import precision
    from models.metrics import recall
    #
    X_train, X_valid, y_train, y_valid = train_test_split(images_main, labels, test_size=0.15,
                                                          random_state=2018)
    logger.info(X_train.shape)
    input_img = Input((128, 128, 13), name='img')
    model = get_unet(input_img, n_filters=13, dropout=0.005, batchnorm=True)

    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[precision, recall, "accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.0001, verbose=1),
        ModelCheckpoint('model_12_11_600.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = model.fit(X_train, y_train, batch_size=8, epochs=100, callbacks=callbacks,
                        validation_data=(X_valid, y_valid))
