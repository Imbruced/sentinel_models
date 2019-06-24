from abc import ABC
from typing import List
from copy import deepcopy

import attr
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from gis import Raster
from gis.raster_components import ReferencedArray
from logs import logger
from exceptions import ConfigException
from gis.enums import ConfDictAttributes


@attr.s
class TrainingConfig(ABC):
    callbacks = attr.ib()
    dropout = attr.ib(default=0.5)
    batchnorm = attr.ib(default=False)
    metrics = attr.ib(default=["accuracy"])
    loss = attr.ib(default="binary_crossentropy")
    optimizer = attr.ib(default=Adam(lr=0.00001))


@attr.s
class EmptyModel(ABC):
    model = attr.ib()

    @staticmethod
    def create(units, activation, **kwags):
        seq = Sequential()
        seq.add(
            Dense(units=units, activation=activation)
        )
        return EmptyModel(
            model=seq
        )

    def build(self):
        pass

    def compile(self):
        pass


@attr.s
class AbstractModel(ABC):

    config = attr.ib()
    is_trained = attr.ib()
    is_compiled = attr.ib()
    model = attr.ib()

    def compile(self):
        self.model.compile(optimizer=self.config.optimizer,
                           loss=self.config.loss,
                           metrics=self.config.metrics
            )
        self.is_compiled = True

    def predict(self, x, threshold) -> Raster:
        logger.info("Walking alone")
        if not self.is_trained:
            raise AttributeError("Model is not trained yet")
        predicted = self.model.predict(np.array([x]))
        filtered = (predicted > threshold).astype(np.uint8)
        raster = Raster.from_array(
            filtered[0, :, :, :],
            pixel=x.pixel,
            extent=x.extent
        )

        return raster

    def fit(self, data, epochs: int):
        self.model.fit(
            np.array(data.x_train),
            np.array(data.y_train),
            batch_size=self.config.batch_size,
            epochs=epochs,
            validation_data=(np.array(data.x_test), np.array(data.y_test)),
            callbacks=self.config.callbacks
        )
        self.is_trained = True

    @classmethod
    def load_from_weight_file(cls, path, config: TrainingConfig):
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


class ModelBuilderConfig(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__validate()

    def __validate_keys(self) -> bool:

        return set(self.keys()).intersection(
            {ConfDictAttributes.at.value,
             ConfDictAttributes.la.value,
             ConfDictAttributes.ind.value
             }
        ).__len__() == len({ConfDictAttributes.at.value,
                            ConfDictAttributes.la.value,
                            ConfDictAttributes.ind.value})

    @classmethod
    def from_yaml(cls):
        pass

    @classmethod
    def from_json(cls):
        pass

    @classmethod
    def from_csv(cls):
        pass

    def __validate_values(self) -> bool:
        if self.values():
            unique_types = set([type(value) for key, value in self.items() if key != ConfDictAttributes.ind.value])
        else:
            return False
        return len(unique_types) == 1 and list(unique_types)[0] == list

    def __validate_sizes(self) -> bool:
        layers = self.get(ConfDictAttributes.la.value, [])
        activation_function = self.get(ConfDictAttributes.at.value, [])

        return len(layers) == len(activation_function) and len(layers) and len(activation_function)

    def __validate(self):
        if not self.__validate_values() or not self.__validate_keys() or not self.__validate_sizes():
            raise ConfigException("Provided bad version of config")

    @property
    def layers(self):
        return self[ConfDictAttributes.la.value]

    @property
    def activations(self):
        return self[ConfDictAttributes.at.value]

    @property
    def input_dim(self):
        return self[ConfDictAttributes.ind.value]


@attr.s
class ModelBuilder(AbstractModel):

    model = attr.ib(default=Sequential())

    def build(self, config: ModelBuilderConfig):
        current_model = self.__add_layer(
            config.layers[0],
            config.activations[0],
            input_dim=config.input_dim
        )

        if len(config.layers) > 1:
            for index in range(1, len(config.layers)):
                print(index)
                current_model = self.__add_layer(
                    config.layers[index],
                    config.activations[index]
            )

        return ModelBuilder(
            model=current_model.__add_layer(
                units=config.layers[-1],
                activations=config.activations[-1]
            ).model,
            config=self.config,
            is_trained=False,
            is_compiled=False
        )

    def __add_layer(self, units: list, activations: list, **kwargs):
        current_seq = self.model

        current_seq.add(Dense(
            units=units,
            activation=activations,
            **kwargs))

        return ModelBuilder(
            model=current_seq.model,
            config=config,
            is_trained=False,
            is_compiled=False
        )

@attr.s
class Ann(ModelBuilder):

    def __attrs_post_init__(self):
        pass

    @classmethod
    def from_dict_config(cls, dict_config: dict):
        """

        :param dict_config:
        :return:
        """


# s = ModelBuilderConfig(
#     activations=["relu", "relu", "relu", "softmax"],
#     layers=[13, 10, 10, 13],
#     input_dim=13
# )
#
# callbacks = [
#     EarlyStopping(patience=100, verbose=1),
#     ReduceLROnPlateau(factor=0.1, patience=100, min_lr=0, verbose=1),
#     ModelCheckpoint('model_more_class_pixels.h5', verbose=1, save_best_only=True, save_weights_only=False)
# ]
#
# config = TrainingConfig(
#     callbacks=callbacks
# )
#
# builder = ModelBuilder(
#     config=config,
#     is_compiled=False,
#     is_trained=False
# )
#
# builder.build(s).compile()
# builder.model.summary()
# # builder.summary()

