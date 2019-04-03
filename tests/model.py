import matplotlib
from gis.geometry import PolygonFrame
from gis.raster import Raster
import numpy as np
import matplotlib.pyplot as plt


matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    shape_path = "D:\\master_thesis\\data\\geometry\\domy.shp"
    polygon_frame = PolygonFrame.from_file(shape_path)
    print([el.wkt for el in polygon_frame.frame["geometry"].values.tolist()])
    # polygon_frame = polygon_frame.union("id")
    main_image = Raster.from_file("D:\\master_thesis\\data\\buildings.tif")
    label_raster = Raster.with_adjustment("from_geo", main_image, polygon_frame)

    plt.imshow(label_raster.array[:, :, 0])
    plt.show()

    raster_2 = Raster.from_wkt("POLYGON ((559610.9241914469 4929878.326498551, 559609.1534925442 4929875.57856005, 559605.5355190277 4929878.42469235, 559607.557427437 4929882.577692465, 559606.1117225888 4929884.705471715, 559613.3150468491 4929894.074087637, 559638.4648143852 4929876.069178509, 559632.0605383542 4929865.969579284, 559630.425247005 4929865.733074644, 559628.0796669627 4929862.315388259, 559625.2011149345 4929865.16829956, 559626.4416073123 4929868.132861895, 559610.9241914469 4929878.326498551))', 'POLYGON ((559649.5178541846 4929868.861598261, 559667.9755624577 4929854.855944711, 559657.7455018568 4929841.47267851, 559654.2344785262 4929844.172108212, 559656.0024547682 4929847.215351267, 559646.6510438476 4929853.18345255, 559644.2438357287 4929850.724975194, 559641.5745055497 4929853.801288999, 559644.1821326618 4929857.442875059, 559642.6375567068 4929858.831445266, 559649.5178541846 4929868.861598261))', 'POLYGON ((559678.1483190297 4929851.479447829, 559702.0260686788 4929833.979895048, 559695.6841651375 4929822.847195794, 559697.3752322081 4929811.271539806, 559668.3094811505 4929807.017622873, 559665.9393011407 4929817.774923258, 559669.4706006899 4929818.619492555, 559669.453642307 4929820.465069342, 559674.4651812499 4929821.175584378, 559674.4360103633 4929824.349976472, 559678.4399469909 4929825.494211257, 559678.8930150989 4929822.17605618, 559690.6509150363 4929824.499002653, 559681.4497492248 4929831.35438757, 559679.4143017627 4929828.677825937, 559676.0109321207 4929831.156745552, 559677.3028687639 4929834.269449122, 559673.6306682228 4929837.262704562, 559674.1378097906 4929839.556073427, 559671.4793145902 4929841.451205881, 559678.1483190297 4929851.479447829))",
                               main_image.extent, main_image.pixel)
    plt.imshow(raster_2.array[:, :, 0])
    plt.show()

    raster_3 = Raster.empty_raster(main_image.extent, main_image.pixel)

    plt.imshow(raster_3.array[:, :, 0])
    plt.show()




    # logger.info(label_raster.array.shape)
    # main_image = Raster.from_file("D:\\master_thesis\\data\\buildings.tif")
    # img_stand = ImageStand(main_image, 3)
    # main_image_stand = img_stand.standarize_image()
    #
    # logger.info("Summary Shapes")
    # logger.info(main_image_stand.array.shape)
    # logger.info(label_raster.array.shape)
    #
    # raster_data = RasterData(main_image_stand.array, label_raster.array)
    # images = raster_data.prepare_unet_images([128, 128])
    #
    # labels = np.array([el[1] for el in images if el[1].shape[0] == 128 and el[1].shape[1] == 128])
    #
    # images = raster_data.prepare_unet_images([128, 128])
    # images_main = np.array([el[0] for el in images if el[0].shape[0] == 128 and el[0].shape[1] == 128])
    #
    # for index, el in enumerate(images_main):
    #     logger.info(el.shape)
    #
    # logger.info(labels.shape)
    # logger.info(images_main.shape)
    # img_stand = None
    # main_image_stand = None
    # label_raster = None
    # main_image = None
    # raster_data = None
    # images = None
    #
    # logger.info("Releasing memory")
    # import time
    # time.sleep(5)
    #
    # #keras part
    # from keras.layers import Input
    # from models.unet import get_unet
    # from keras.optimizers import Adam
    # from models.metrics import precision
    # from models.metrics import recall
    # #
    # logger.info(f"Images main shape line 68 {images_main.shape}")
    # logger.info(f"Label shape lines 69 {labels.shape}")
    # X_train, X_valid, y_train, y_valid = train_test_split(images_main, labels, test_size=0.15,
    #                                                       random_state=2018)
    # logger.info(X_train.shape)
    # input_img = Input((128, 128, 3), name='img')
    # model = get_unet(input_img, n_filters=13, dropout=0.05, batchnorm=True)
    #
    # model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[precision, recall, "accuracy"])
    # model.summary()
    #
    # callbacks = [
    #     EarlyStopping(patience=10, verbose=1),
    #     ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.0001, verbose=1),
    #     ModelCheckpoint('model_12_11_600.h5', verbose=1, save_best_only=True, save_weights_only=True)
    # ]
    # logger.info(X_train.shape)
    # logger.info(y_train.shape)
    # results = model.fit(X_train, y_train, batch_size=2, epochs=1, callbacks=callbacks,
    #                     validation_data=(X_valid, y_valid))
    #
    # preds_train = model.predict(X_train)
    # preds_val = model.predict(X_valid)
    #
    # preds_train_t = (preds_train > 0.5).astype(np.uint8)
    # preds_val_t = (preds_val > 0.5).astype(np.uint8)
    #
    # import random
    #
    # logger.info(preds_val_t[0, :, :, :].shape)
    # logger.info(preds_train_t[0, :, :, :].shape)
    #
    # plt.imshow(preds_val_t[0, :, :, 0])
    # plt.show()
    # plt.imshow(X_train[0, :, :, 0])

