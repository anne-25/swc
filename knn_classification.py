from svm_classification import dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_knn(X_train, X_test, y_train, y_test):

    neigh  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
    neigh.fit(X_train, y_train)
    print(neigh.score(X_train, y_train))
    print(neigh.score(X_test, y_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = dataset()

    train_knn(X_train, X_test, y_train, y_test)
