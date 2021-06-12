import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, metrics, preprocessing, datasets, model_selection
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

#datasets
df0 = pd.read_csv('data/point_dist_kpts.csv', header=None)
df0[20] = 0

df1 = pd.read_csv('rock/new_dist_kpts.csv', header=None)
df1[20] = 1

df2 = pd.read_csv('scissors/new_dist_kpts.csv', header=None)
df2[20] = 2

df = pd.concat([df0, df1, df2], ignore_index=True)

X = df.iloc[:, 0:20]
y = df.iloc[:, [20]]
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# estimator = SVC(C=50, kernel='rbf', gamma=0.01)

clf = OneVsRestClassifier(SVC())

parameters = {
    'estimator__C': np.logspace(-4, 4, 5),
    'estimator__gamma': np.logspace(-4, 4, 5),
}

model = GridSearchCV(
    estimator = clf,
    param_grid = parameters,
    cv = 4,
    verbose = 2
)

model.fit(X_train, y_train)

result = pd.DataFrame.from_dict(model.cv_results_)
result.to_csv('result.csv')

best = model.best_estimator_
pred = best.predict(X_test)

print(metrics.confusion_matrix(y_test, pred))

# pred_y = clf.predict(X_test)
# print ('One-versus-the-rest: {:.5f}'.format(accuracy_score(y_test, pred_y)))
