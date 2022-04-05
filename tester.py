import source.plotting
import source.data_handling
import source.descriptive_statistics
import source.diy_classifiers
import numpy as np

data_version = source.data_handling.GENRE_CLASS_DATA_30S
data_set = source.data_handling.read_genre_class_data(data_version)

features = source.mappings.MUSIC_FEATURES_ALL

training_data, test_data = source.data_handling.prepare_data(
        data_frame=data_set,
        features=features,
    )
print(len(features))
print(training_data.y)
print(type(data_set))


k = 5
features_task_1 = ["spectral_rolloff_mean", "tempo", "spectral_centroid_mean", "mfcc_1_mean"]

ks = np.arange(10)*3 + 1
ps = np.arange(10) + 1

print(ks)
print(ps)


y_pred_task1 = source.diy_classifiers.kNN(k, training_data, test_data, p=2)
y_true_task1 = test_data.y

#source.plotting.confusion_matrix(y_true_task1, y_pred_task1)
source.plotting.error_rates_vs_params(ks, ps, training_data, test_data)

#principal_components_reduction(X, genres)
#threed_scatter(X, features=features, genres=genres)

