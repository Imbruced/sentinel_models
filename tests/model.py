import matplotlib
from gis.geometry import PolygonFrame
from gis.raster import Raster
from logs.log_lib import logger
from gis.standarizer import ImageStand
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from gis.data_prep import RasterData
from models import Unet
from models import UnetConfig
from keras.optimizers import SGD
from pprint import pprint
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from plotting import ImagePlot

matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    shape_path = "D:\\master_thesis\\PYTHON\\sentinel_models\\tests\\domy.shp"
    polygon_frame = PolygonFrame.from_file(shape_path).union("id")
    logger.info(polygon_frame.crs)
    main_image = Raster.from_file("D:\\master_thesis\\PYTHON\\sentinel_models\\tests\\buildings.tif", crs="epsg:26917")
    label_raster = Raster.with_adjustment("from_geo", main_image, polygon_frame)
    stand_scaler = StandardScaler()
    # max_scaler = MaxScaler()
    standarize1 = ImageStand(raster=main_image)
    # standarize2 = ImageStand(raster=main_image)
    raster_data = RasterData(standarize1.standarize_image(stand_scaler), label_raster)
    unet_images = raster_data.prepare_unet_images(image_size=(64, 64))
    callbacks = [
        EarlyStopping(patience=100, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=100, min_lr=0, verbose=1),
        ModelCheckpoint('model_more_class_pixels.h5', verbose=1, save_best_only=True, save_weights_only=False)
    ]
    config = UnetConfig(
        input_size=[64, 64, 3]
    )
    unet = Unet(config=config)
    # unet.compile(metrics=["accuracy"], optimizer=SGD(lr=0.001))
    # unet.fit(unet_images, callbacks=callbacks, epochs=400, batch_size=2)
    model_path = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\model_more_class_pixels.h5"
    unet = Unet.load_from_weight_file(model_path, config=config)
    # logger.info(unet.config.model)
    # logger.info(type(unet))
    predicted = unet.predict(unet_images.x_test[:10, :, :, :], threshold=0.6)

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

    # raster_data.create_ann_frame().to_csv("C:\\Users\\Pawel\\Desktop\\sentinel_models\\example.csv")
    # plt.imshow(raster_data.create_ann_frame().reshape(1677, 1673, 4)[:, :, :3])
    # plt.show()
    # plt.imshow(main_image.array[:, :, :3])
    # plt.show()
    # raster_data.save_unet_images([128, 128], "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests")

