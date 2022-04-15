import numpy as np
import pandas
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import sklearn.model_selection

import source.data_handling
import source.mappings
import source.plotting


class MLP:
    def __init__(self, input_dimensions=None, output_dimensions=None, model=None, verbose=True):
        if model is None and input_dimensions is None and output_dimensions is None:
            raise KeyError("Either dimensions or model has to be given")
        
        if model is None:
            first_layer_dimensions = np.max([input_dimensions // 2, output_dimensions + 4])
            second_layer_dimensions = np.max([input_dimensions // 4, output_dimensions + 2])
            self.model = keras.Sequential([
                keras.layers.Dense(input_dimensions, activation=tf.nn.relu, input_dim=input_dimensions),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(first_layer_dimensions, activation=tf.nn.relu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(second_layer_dimensions, activation=tf.nn.relu),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(output_dimensions, activation=tf.nn.softmax),
            ])
        else:
            self.model = model

        self.model.summary()
        self._verbose = verbose

    def fit(
        self,
        training_data_x: pandas.DataFrame,
        training_data_y: pandas.DataFrame,
    ):
        # preprocess data
        self._categories = list(set(training_data_y))

        labels = [self._categories.index(category) for category in training_data_y]
        categorical_y = keras.utils.to_categorical(labels, len(self._categories))

        skf = sklearn.model_selection.StratifiedKFold(n_splits=6)
        for train_indices, validation_indices in skf.split(training_data_x, training_data_y):
            pass
        
        self._means = np.mean(training_data_x)
        training_data_x = training_data_x - self._means

        train_x = training_data_x.iloc[train_indices]
        train_y = categorical_y[train_indices]
        valid_x = training_data_x.iloc[validation_indices]
        valid_y = categorical_y[validation_indices]

        # training model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self._history = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=1000,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=self._verbose,
        )

    def predict(
        self,
        test_data_x: pandas.DataFrame,
    ):
        # preprocessing data
        test_x = test_data_x - self._means
        y_predicted_probabilities = self.model.predict(test_x)

        y_predicted_genres = [self._categories[np.argmax(prediction)] for prediction in y_predicted_probabilities]

        return y_predicted_genres

