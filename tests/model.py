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
    # callbacks = [
    #     EarlyStopping(patience=10, verbose=1),
    #     ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.0001, verbose=1),
    #     ModelCheckpoint('model_12_11_600.h5', verbose=1, save_best_only=True, save_weights_only=False)
    # ]
    config = UnetConfig(
        input_size=[64, 64, 3]
    )
    # unet = Unet(config=config)
    # unet.compile(metrics=["accuracy"])
    # unet.fit(unet_images, callbacks=callbacks, epochs=60, batch_size=8)
    model_path = "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests\\model_12_11_600.h5"
    unet = Unet.load_from_weight_file(model_path, config=config)
    # logger.info(unet.config.model)
    # logger.info(type(unet))
    predicted = unet.predict(unet_images.x_test[:1, :, :, :], threshold=0.7)

    image_plot = ImagePlot()
    image_plot.extend(
        [
            unet_images.x_test[0, :, :, :3],
            unet_images.y_test[0, :, :, 0],
            predicted[0, :, :, 0]
        ]
    )
    image_plot.plot()

    # raster_data.create_ann_frame().to_csv("C:\\Users\\Pawel\\Desktop\\sentinel_models\\example.csv")
    # plt.imshow(raster_data.create_ann_frame().reshape(1677, 1673, 4)[:, :, :3])
    # plt.show()
    # plt.imshow(main_image.array[:, :, :3])
    # plt.show()
    # raster_data.save_unet_images([128, 128], "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests")

