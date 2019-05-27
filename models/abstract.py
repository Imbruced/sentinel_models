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

from logs import logger
from exceptions import ConfigException
from gis.enums import ConfDictAttributes


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
    x = attr.ib(type=np.ndarray)
    y = attr.ib(type=np.ndarray)
    test_size = attr.ib(default=0.15)
    random_state = attr.ib(default=2018)

    def __attrs_post_init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.number_of_classes = np.unique(self.y)


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

empty_model = EmptyModel.create(10, "relu")
empty_model.model.summary()


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

    def predict(self, x, threshold) -> np.ndarray:
        logger.info("Walking alone")
        if not self.is_trained:
            raise AttributeError("Model is not trained yet")
        predicted = self.model.predict(x)

        return (predicted > threshold).astype(np.uint8)

    def fit(self, data: ModelData, epochs: int):
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


class ConfigDict(dict):

    def __init__(self, **kwargs):
        """
        TODO add enumeration
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.__activation_name = "activation"
        self.__layers = "layers"
        self.__input_dim = "input_dim"
        self.__validate()

    def __validate_keys(self) -> bool:
        """
        TODO
        enumeration should have proper length method to simplify the code in ine 125
        :return:
        """
        return set(self.keys()).intersection(
            {self.__activation_name,
             self.__layers,
             self.__input_dim
             }
        ).__len__() == len({self.__activation_name, self.__layers, self.__input_dim})

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
            unique_types = set([type(value) for key, value in self.items() if key != "input_dim"])
        else:
            return False
        return len(unique_types) == 1 and list(unique_types)[0] == list

    def __validate_sizes(self) -> bool:
        layers = self.get(self.__layers, [])
        activation_function = self.get(self.__activation_name, [])

        return len(layers) == len(activation_function) and len(layers) and len(activation_function)

    def __validate(self):
        if not self.__validate_values() or not self.__validate_keys() or not self.__validate_sizes():
            raise ConfigException("Provided bad version of config")


@attr.s
class ModelBuilder(AbstractModel):

    config_dict = attr.ib(type=ConfigDict)
    model = attr.ib(default=Sequential())

    def build(self):
        current_model = self.__add_layer(
            self.config_dict["layers"][0],
            self.config_dict["activation"][0],
            input_dim=self.config_dict["input_dim"]
        )

        if len(self.config_dict["layers"]) > 1:
            for index in range(1, len(self.config_dict["layers"])):
                print(index)
                current_model = self.__add_layer(
                    self.config_dict["layers"][index],
                    self.config_dict["activation"][index]
            )

        return ModelBuilder(
            model=current_model.__add_layer(
                units=self.config_dict["layers"][-1],
                activation=self.config_dict["activation"][-1]
            ).model,
            config=self.config,
            config_dict=self.config_dict,
            is_trained=False,
            is_compiled=False
        )

    def __add_layer(self, units, activation, **kwargs):
        current_seq = self.model

        current_seq.add(Dense(
            units=units,
            activation=activation,
            **kwargs))

        return ModelBuilder(
            model=current_seq.model,
            config_dict=deepcopy(self.config_dict),
            config=self.config,
            is_trained=False,
            is_compiled=False
        )




@attr.s
class AnnConfig(ModelConfig):
    pass


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


s = ConfigDict(
    activation=["relu", "relu", "relu", "softmax"],
    layers=[13, 10, 10, 13],
    input_dim=13
)

callbacks = [
    EarlyStopping(patience=100, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=100, min_lr=0, verbose=1),
    ModelCheckpoint('model_more_class_pixels.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

config = ModelConfig(
    callbacks=callbacks
)

builder = ModelBuilder(
    config_dict=s,
    config=config,
    is_compiled=False,
    is_trained=False
)

builder.build().compile()
builder.summary()

