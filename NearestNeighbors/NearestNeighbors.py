import numpy as np
import pandas as pd

def import_data(file):
    # Read data from CSV file
    data = pd.read_csv(file, index_col='Index')
    data = np.asarray(data.values)
    return data

def euclidean_distance(x, y):
    # Calculate distance using euclidean distance formula
    distance = np.sqrt(np.sum(np.power((x - y), 2)))
    return distance

def manhattan_distance(x, y):
    # Calculate distance using manhattan distance formula
    distance = abs(np.sum(x - y))
    return distance

def random_data(array):
    # Random index of data
    num = len(array)
    shuffle_index = np.random.permutation(num)
    data_shuffle = array[shuffle_index]
    return data_shuffle

def accuracy(x, y):
    # Compute accuracy between prediction and training data
    array_score = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(y)):
            if (x[i] == y[i]):
                array_score[i] = 1
            else:
                array_score[i] = 0

    score = sum(array_score) / len(x)
    return score

def KNearestNeighbors(data_train, data_validation, label_train, k, distance='euclidean'):
    data_distance = []
    predict = []
    label = np.zeros(4)
    for i in range(len(data_validation)):
        for j in range(len(data_train)):
            if distance == 'euclidean':
                euclidean = euclidean_distance(data_train[j], data_validation[i])
                data_distance.append(euclidean)
            elif distance == 'manhattan':
                manhattan = manhattan_distance(data_train[j], data_validation[i])
                data_distance.append(manhattan)

            if (len(data_distance) == len(data_train)):
                data_distance = np.asarray(data_distance)
                sorted_index = np.argsort(data_distance)
                data_predict = data_distance[sorted_index]
                label_predict = label_train[sorted_index]
                label_voting = label_predict[:k]

                # TODO: Fix voting system
                for k in range(len(label_voting)):
                    if (label_voting[k] == 0):
                        label[0] += 1
                    elif (label_voting[k] == 1):
                        label[1] += 1
                    elif (label_voting[k] == 2):
                        label[2] += 1
                    elif (label_voting[k] == 3):
                        label[3] += 1
                max_index = np.argmax(label)
                predict.append(max_index)
                data_distance = []

    return predict

def NearestNeighbors(data_train, data_validation, label_train, distance='euclidean'):
    data_distance = []
    predict = []
    for i in range(len(data_validation)):
        for j in range(len(data_train)):
            if distance == 'euclidean':
                euclidean = euclidean_distance(data_train[j], data_validation[i])
                data_distance.append(euclidean)
            elif distance == 'manhattan':
                manhattan = manhattan_distance(data_train[j], data_validation[i])
                data_distance.append(manhattan)

            if (len(data_distance) == len(data_train)):
                data_distance = np.asarray(data_distance)
                sorted_index = np.argsort(data_distance)
                data_predict = data_distance[sorted_index]
                label_predict = label_train[sorted_index]
                predict.append(label_predict[0])
                data_distance = []

    return predict

def main():
    # import data
    data = import_data('data/data_train.csv')
    data = random_data(data)

    # separate data and label
    data_train, label_train = data[:600,0:5], data[:600,5]
    data_validation, label_validation = data[600:800,0:5], data[600:800,5]

    # perform Nearest Neighbors algorithm
    # predict = NearestNeighbors(data_train, data_validation, label_train)
    predict = KNearestNeighbors(data_train, data_validation, label_train, 11)

    # calculate prediction accuracy
    score = accuracy(predict, label_validation)
    score = score * 100
    print('Akurasi : %s' %score)

if __name__ == '__main__':
    main()