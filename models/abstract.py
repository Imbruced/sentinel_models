from abc import ABC
from typing import List

import attr
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam

from logs import logger


@attr.s
class ModelConfig(ABC):
    callbacks = attr.ib()
    dropout = attr.ib(default=0.5)
    batchnorm = attr.ib(default=False)
    metrics = attr.ib(default=["accuracy"])
    loss = attr.ib(default="binary_crossentropy")
    optimizer = attr.ib(default=Adam(lr=0.00001))


@attr.s
class ModelData(ABC):
    x = attr.ib()
    y = attr.ib()
    test_size = attr.ib(default=0.15)
    random_state = attr.ib(default=2018)

    def __attrs_post_init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )


@attr.s
class AbstractModel(ABC):

    config = attr.ib()
    is_trained = attr.ib(default=False, init=False)
    is_compiled = attr.ib(default=False, init=False)
    model = attr.ib(default=None)

    def compile(self):
        self.config.model.compile(optimizer=self.config.optimizer,
                                  loss=self.loss,
                                  metrics=self.metrics
                                  )
        self.is_compiled = True

    def predict(self, x, threshold) -> np.ndarray:
        logger.info("Walking alone")
        if not self.is_trained:
            raise AttributeError("Model is not trained yet")
        predicted = self.model.predict(x)

        return (predicted > threshold).astype(np.uint8)

    def fit(self, data, epochs):
        self.config.model.fit(
            data.x_train,
            data.y_train,
            batch_size=self.config.batch_size,
            epochs=epochs,
            validation_data=(data.x_test, data.y_test),
            callbacks=self.config.callbacks
        )
        self.is_trained = True

    @classmethod
    def load_from_weight_file(cls, path, config: ModelConfig):
        """

        :param path:
        :param config:
        :return:
        """
        model = load_model(path)
        new_model = cls(config=config)
        new_model.model = model
        new_model.is_trained = True
        new_model.is_compiled = True
        return new_model

    def summary(self):

        if self.is_compiled:
            self.config.model.summary()
        else:
            logger.info("Model is not compiled, please compile it")




