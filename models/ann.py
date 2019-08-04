import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from accuracy_functions import fmeasure
from sklearn.preprocessing import StandardScaler
from accuracy_functions import precision
from accuracy_functions import recall
import itertools


def return_max_label(*args, columns):
    max_value = max(args)

    index_max = args.index(max_value) if max_value >= 0.51 else -1

    try:
        res = columns[index_max]
    except IndexError:
        res = 0

    return int(res)


class InputFrame:

    def __init__(self, input_path, *args, **kwargs):
        self.__input_path = input_path
        self.__input_frame = self.load_file(sep=kwargs.get("sep", "|"))
        self.__aligned_data = self.align_data(values=kwargs.get("values", 600),
                                              label_columns=kwargs.get("label_column", "label"))
        self.standard_scaler = StandardScaler()
        self.train_set = self.train_test_split(**kwargs)[0:2]
        self.test_set = self.train_test_split(**kwargs)[2:]

    def find_hidden_layers_number(self, coeff=3, **kwargs):
        hidden_layers_params = self.hiddenl_params(**kwargs)
        hiddenl_numb = int(hidden_layers_params[2] / ((hidden_layers_params[1] + hidden_layers_params[0]) * coeff))

        return hiddenl_numb

    def hiddenl_params(self, **kwargs):
        unique_values = self.find_unique_label_values(**kwargs)
        assert unique_values.__len__() < 200

        input_dim, feature_nr = self.train_set[0].shape[1] - 1, self.train_set[0].shape[0]

        return float(unique_values.__len__()), float(input_dim), float(feature_nr)

    def scale_data(self, **kwargs):
        label_column = kwargs.get("label_column", "label")

        whole_columns = self.__aligned_data.columns.tolist()

        whole_columns.remove(label_column)

        inputs = pd.DataFrame(self.standard_scaler.fit_transform(self.__aligned_data[whole_columns]))
        label = pd.get_dummies(self.__aligned_data[label_column])

        return inputs, label

    def align_data(self, **kwargs):
        unique_values = self.find_unique_label_values(**kwargs)
        label_column = kwargs.get("label_column", "label")

        frames = []
        for x in unique_values:
            frames.append(self.__input_frame[self.__input_frame[label_column] == x].iloc[:kwargs.get("values", 600)])

        res = pd.concat([*frames], axis=0)

        return pd.concat([*frames], axis=0)

    def find_unique_label_values(self, **kwargs):
        label_column = kwargs.get("label_column", "label")
        assert isinstance(label_column, str)

        unique_values = self.__input_frame[label_column].unique().tolist()

        return unique_values


class Model:

    def __init__(self, input_table, **kwargs):
        self.model = Sequential()
        self.input_dim = kwargs.get("input_dim", 13)
        self.hidden_layers_number = kwargs.get("hidden_layers_number", 3)
        self.units = kwargs.get("units", 10)
        self.activation = kwargs.get("activation", "relu")
        self.number_of_classes = kwargs.get("number_of_classes", 15)
        self.name = kwargs.get("name", None)
        self.callbacks = [
            EarlyStopping(patience=10, verbose=1, monitor="loss", mode='auto'),
            ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.01, verbose=1, monitor="loss"),
            ModelCheckpoint(f'{self.name}.h5', verbose=1, save_best_only=True, save_weights_only=True, monitor="loss")
        ]
        self.input_table = input_table

    def validate_image(self):
        tests = pd.read_csv("D:\\master_thesis\\data\\20160616\\test\\csv\\test_data_3.csv", sep="|")
        inputs_test = pd.DataFrame(self.input_table.standard_scaler.fit_transform(tests)).values
        columns = self.input_table.train_set[1].columns
        predicted_test = pd.DataFrame(self.model.predict(inputs_test),
                                      columns=columns).apply(lambda x: return_max_label(*x, columns=columns),
                                                             axis=1).values

        data = predicted_test.reshape(248, 331)

        return data

    def validate_test_set(self):
        pass

    def fit_model(self, **kwargs):
        self.build_model(**kwargs)
        self.model.fit(self.input_table.train_set[0],
                       self.input_table.train_set[1],
                       epochs=500,
                       batch_size=2,
                       callbacks=self.callbacks)

    def build_model(self, **kwargs):

        self.add_hidden_layers(**kwargs)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer="Adam",
                           metrics=[precision, recall, "accuracy", fmeasure])

    def add_hidden_layers(self, **kwargs):

        self.model.add(Dense(units=self.units,
                             activation=self.activation,
                             input_dim=self.input_dim))

        if self.hidden_layers_number > 2:
            for hiden_layer in range(self.hidden_layers_number - 2):
                self.model.add(Dense(units=self.units,
                                     activation=self.activation))

        self.model.add(Dense(units=self.number_of_classes, activation='softmax'))

    @property
    def hidden_layers_number(self):
        return self.__hidden_layers_number

    @hidden_layers_number.setter
    def hidden_layers_number(self, value):
        if value <= 1:
            raise AttributeError("Value must be more or equal than 1")
        else:
            self.__hidden_layers_number = value

    @property
    def activation(self):
        return self.__activation

    @activation.setter
    def activation(self, value):
        if value not in ["relu", "sigmoid", "tanh"]:
            raise AttributeError("""activation function must be in ["relu", "sigmoid", "tanh"]""")
        else:
            self.__activation = value


class GridSearch:

    def __init__(self, input_table: InputFrame, **kwargs):
        self.result_images = dict()
        self.__test_results = []
        self.__input_table = input_table
        self.params = dict(
            activation=["sigmoid", "tanh", "relu"],
            hidden_layers=list(range(3, 7)),
            unit=[8, 16, 32, 64, 128, 256]
            #             [self.__input_table.find_hidden_layers_number(coeff=x) for x in range(2, 10)]
        )

    def search(self):
        for el in itertools.product(*list(self.params.values())):
            self.model = Model(input_table=self.__input_table, name="_".join([str(x) for x in el]))
            print(f"""{el}""")
            self.model.activation = el[0]
            self.model.hidden_layers_number = el[1]
            self.unit = el[2]
            self.model.fit_model()
            self.result_images[self.model.name] = self.model.validate_image()
