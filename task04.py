from pyexpat import features
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
import matplotlib.pyplot as plt
import copy

def validate_plsr_da(training_set):
    """
    Loops over truncation order for the PCA model and computes cross-validated PIs for each truncation order with the PLSR-DA method.
    """
    nr_components = np.shape(training_set.x)[1]
    principle_components_PIs = np.empty((nr_components, ))
    for i in range(nr_components):

        PLSR_DA = source.diy_classifiers.PLSR_DA(i+1)
        PI = source.descriptive_statistics.cross_validate(training_x=training_set.x, training_y=training_set.y, nr_segments=5, classifier=PLSR_DA)
        principle_components_PIs[i] = PI

    return principle_components_PIs

if __name__ == "__main__":
    data_version = source.data_handling.GENRE_CLASS_DATA_30S
    data_frame = source.data_handling.read_genre_class_data(data_version)

    scaled_data_frame = copy.deepcopy(data_frame)
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(data_frame[source.mappings.MUSIC_FEATURES_ALL])

    scaled_data_frame[source.mappings.MUSIC_FEATURES_ALL] = scaler.transform(scaled_data_frame[source.mappings.MUSIC_FEATURES_ALL])

    genres = list(source.mappings.GENRES.values())

    training_data, test_data = source.data_handling.prepare_data(
                data_frame=scaled_data_frame,
                features=source.mappings.MUSIC_FEATURES_ALL,
            )

    # Model order selection for PCA


    CV_vars = source.descriptive_statistics.cross_validate_pca(training_x=training_data.x, training_y=training_data.y, nr_segments=5)
    Scores, P, singular_values, cum_var = source.descriptive_statistics.perform_PCA(training_data.x)
    nr_features = np.shape(cum_var)[0]
    x_axis = np.linspace(1, nr_features, nr_features)

    fig0 = plt.figure()
    plt.title("Explained Variance Plots")
    plt.plot(x_axis, cum_var, label="Explained Variance")
    plt.plot(x_axis, CV_vars, label="Explained Variance through Cross-Validation")
    plt.xlabel("PCA/SVD Truncation Order")
    plt.ylabel("Fraction of Explained Variance")
    plt.grid()

    plt.legend()
    plt.show()

    CV = False
    if CV:
        principle_components_PIs = validate_plsr_da(training_set=training_data)


        fig0 = plt.figure()
        plt.title("Error Rate as a Function of Principle Component Nr")
        plt.plot(x_axis, principle_components_PIs, label="Cross Validation Error Rate for the PLSR-DA Classifier")
        plt.xlabel("PCA/SVD Truncation Order")
        plt.ylabel("Error Rate [%]")
        plt.grid()

        plt.legend()
        plt.show()

    features = source.mappings.MUSIC_FEATURES_ALL

    #fig = plt.figure()
    #ax = fig.add_subplot()
    #ax.set_xlabel("Component 1")
    #ax.set_ylabel("Component 2")
    #ax.grid()
    #
    #for i in range(len(features)):
    #    ax.scatter(P[0,i], P[1,i], label=features[i])
    #plt.legend()
    #plt.show()


    # PLSR_DA
    myClassifier = source.diy_classifiers.PLSR_DA(n_components=20)
    myClassifier.fit(training_data.x, training_data.y)

    y_genres_pred = myClassifier.predict(test_data=test_data.x)
    source.plotting.confusion_matrix(test_data.y, y_genres_pred)