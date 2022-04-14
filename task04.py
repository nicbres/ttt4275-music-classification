import sklearn
import source.data_handling
import source.mappings
import source.plotting
import source.diy_classifiers
import source.descriptive_statistics
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

import numpy as np
import pandas as pd
import copy

data_version = source.data_handling.GENRE_CLASS_DATA_30S
data_frame = source.data_handling.read_genre_class_data(data_version)

scaled_data_frame = copy.deepcopy(data_frame)
scaler = sklearn.preprocessing.MaxAbsScaler()
scaler.fit(data_frame[source.mappings.MUSIC_FEATURES_ALL])

scaled_data_frame[source.mappings.MUSIC_FEATURES_ALL] = scaler.transform(scaled_data_frame[source.mappings.MUSIC_FEATURES_ALL])

genres = list(source.mappings.GENRES.values())

training_data, test_data = source.data_handling.prepare_data(
            data_frame=scaled_data_frame,
            features=source.mappings.MUSIC_FEATURES_ALL,
        )

myClassifier = source.diy_classifiers.PLSR_DA(n_components=30)
myClassifier.fit(training_data.x, training_data.y)

y_genres_pred = myClassifier.predict(test_data=test_data.x)

source.plotting.confusion_matrix(test_data.y, y_genres_pred)