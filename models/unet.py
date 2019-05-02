import attr
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from logs import logger
from metrics import precision
from metrics import recall


def conv2d_block(input_tensor, n_filters, kernel_size, batchnorm):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters, dropout, batchnorm):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='relu')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


@attr.s
class UnetConfig:

    input_size = attr.ib(default=(128, 128, 13))
    filters = attr.ib(default=16)
    dropout = attr.ib(default=0.5)
    batchnorm = attr.ib(default=False)

    def __attrs_post_init__(self):
        self.input_image = Input(self.input_size, name='img')
        self.model = get_unet(self.input_image,
                              self.filters,
                              self.dropout,
                              self.batchnorm)


@attr.s
class UnetData:
    """
    TODO
    add validators
    """

    x = attr.ib()
    y = attr.ib()
    test_size = attr.ib(default=0.15)
    random_state = attr.ib(default=2018)

    def __attrs_post_init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = self.__split_data()

    def __split_data(self):
        return train_test_split(
            self.x,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )


@attr.s
class Unet:

    is_trained = attr.ib(default=False, init=False)
    config = attr.ib(default=UnetConfig())
    is_compiled = attr.ib(default=False, init=False)

    def fit(self, data, callbacks, batch_size, epochs):
        self.config.model.fit(
            data.x_train,
            data.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(data.x_test, data.y_test),
            callbacks=callbacks
        )
        self.is_trained = True

    def compile(self, optimizer=Adam(), loss="binary_crossentropy", metrics=[precision, recall, "accuracy"], **kwargs):
        self.config.model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=metrics,
                                  **kwargs)
        self.is_compiled = True

    def save(self, path):
        pass

    def predict(self, x):
        if not self.is_trained:
            raise AttributeError("Model is not trained yet")
        return self.config.model.predict(x)

    @classmethod
    def load_from_file(cls, path):
        """
        This method is created to load model wages from existing file
        :param path: location of model file
        :return:
        """
        pass

    @classmethod
    def from_rasters(cls, unet_config: UnetConfig):
        pass

    @classmethod
    def from_arrays(cls, unet_co):
        pass

    def summary(self):

        if self.is_compiled:
            self.config.model.summary()
        else:
            logger.info("Model is not compiled, please compile it")

