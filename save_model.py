import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, metrics, preprocessing, datasets, model_selection
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
import pickle

#datasets
def dataset():
    df0 = pd.read_csv('data/point/new_dist_kpts.csv', header=None)
    df0[20] = 0

    df1 = pd.read_csv('data/rock/new_dist_kpts.csv', header=None)
    df1[20] = 1

    df2 = pd.read_csv('data/scissors/new_dist_kpts.csv', header=None)
    df2[20] = 2

    df3 = pd.read_csv('data/paper/new_dist_kpts.csv', header=None)
    df3[20] = 3

    df4 = pd.read_csv('data/call/new_dist_kpts.csv', header=None)
    df4[20] = 4

    df5 = pd.read_csv('data/cylinder/new_dist_kpts.csv', header=None)
    df5[20] = 5

    df6 = pd.read_csv('data/good/new_dist_kpts.csv', header=None)
    df6[20] = 6

    df7 = pd.read_csv('data/ok/new_dist_kpts.csv', header=None)
    df7[20] = 7

    df8 = pd.read_csv('data/three/new_dist_kpts.csv', header=None)
    df8[20] = 8

    df9 = pd.read_csv('data/cross/new_dist_kpts.csv', header=None)
    df9[20] = 9

    df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)

    X = df.iloc[:, 0:20]
    y = df.iloc[:, [20]]
    X = np.array(X)
    y = np.array(y)


    y = np.ravel(y)

    return X, y


def train_save_knn(X, y):

    neigh  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
    neigh.fit(X, y)
    filename = '/model/knn_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def train_save_svc(X, y):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    filename = '/model/svc_model.sav'
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    X, y = dataset()
    train_save_svc(X_train, X_test, y_train, y_test)
