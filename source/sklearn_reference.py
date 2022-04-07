import pandas as pd
import sklearn.neighbors
import sklearn.svm

import source.data_handling as dh


class KNN:
    @staticmethod
    def predict(
        training_data: dh.Dataset,
        test_data: dh.Dataset,
        k=5,
        p=2
    ) -> pd.DataFrame:
        knn_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, p=p)
        knn_classifier.fit(X=training_data.x, y=training_data.y)

        return knn_classifier.predict(X=test_data.x)


class SVM:
    @staticmethod
    def predict(
        training_data: dh.Dataset,
        test_data: dh.Dataset,
    ):
        svm_classifier = sklearn.svm.SVC(kernel='rbf', decision_function_shape='ovo')
        svm_classifier.fit(training_data.x, training_data.y)

        return svm_classifier.predict(test_data.x)
