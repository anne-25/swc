import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    fin = open(filename, 'r')

    data = []
    while True:
        line = fin.readline()
        if line == '': break
        line = line.split(',')
        line = [float(el) for el in line]
        data.append(line)
    data = np.array(data)

    return data

def train_knn(datas, labels):

    X_point = datas[0][:400]
    X_point_test = datas[0][400:]

    X_rock = datas[1][:160]
    X_rock_test = datas[1][160:]

    X_scissors = datas[2][:140]
    X_scissors_test = datas[2][140:]



    X = np.concatenate([X_point, X_rock, X_scissors], axis = 0)
    y = np.concatenate([[labels[0]] * X_point.shape[0],
                        [labels[1]] * X_rock.shape[0],
                        [labels[2]] * X_scissors.shape[0]])

    X_test = np.concatenate([X_point_test, X_rock_test, X_scissors_test], axis = 0)
    y_test = np.concatenate([[labels[0]] * X_point_test.shape[0],
                        [labels[1]] * X_rock_test.shape[0],
                        [labels[2]] * X_scissors_test.shape[0]])

    print('training X:', X.shape)
    print('training y:', y.shape)

    print('testing X:', X_test.shape)
    print('testing y:', y_test.shape)

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X, y)
    print(neigh.score(X, y))
    print(neigh.score(X_test, y_test))


    

def train_svc(datas, labels):

    #
    # X_point = datas[0][:400]
    # X_point_test = datas[0][400:]
    #
    # X_rock = datas[1][:160]
    # X_rock_test = datas[1][160:]
    #
    # X_scissors = datas[2][:140]
    # X_scissors_test = datas[2][140:]

    X = np.concatenate([X_point, X_rock, X_scissors], axis = 0)
    y = np.concatenate([[labels[0]] * X_point.shape[0],
                        [labels[1]] * X_rock.shape[0],
                        [labels[2]] * X_scissors.shape[0]])

    X_test = np.concatenate([X_point_test, X_rock_test, X_scissors_test], axis = 0)
    y_test = np.concatenate([[labels[0]] * X_point_test.shape[0],
                        [labels[1]] * X_rock_test.shape[0],
                        [labels[2]] * X_scissors_test.shape[0]])

    print('training X:', X.shape)
    print('training y:', y.shape)

    print('testing X:', X_test.shape)
    print('testing y:', y_test.shape)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    print(clf.score(X, y))
    print(clf.score(X_test, y_test))


    pass


if __name__ == '__main__':

    filefolders =  ['point/', 'rock/', 'scissors/', 'paper/', 'call/', 'ok/', 'good/', 'cross/']
    filename = 'new_dist_kpts.csv'

    datas = []
    for filefolder in filefolders:
        filepath = 'data/' + filefolder + filename
        data = load_data(filepath)
        datas.append(data)

    datas_cleaned = []
    for data in datas:
        if data.shape[1] == 21:
            data = data[:,1:]
        datas_cleaned.append(data)
        #print(data.shape)
    #point = 0, rock = 1, scissors = 2, paper = 3, call = 4, ok = 5, good = 6, cross = 7
    labels = [0, 1, 2, 3, 4, 5, 6, 7]

    #train_knn(datas_cleaned, labels)
    train_svc(datas_cleaned, labels)
