from NearestNeighbors import NearestNeighbors

# import data
data = NearestNeighbors.import_data('data/data_train.csv')
data = NearestNeighbors.random_data(data)
# separate data and label
data_train, label_train = data[:600,0:5], data[:600,5]
data_validation, label_validation = data[600:800,0:5], data[600:800,5]
# perform Nearest Neighbors algorithm
predict = NearestNeighbors.NearestNeighbors(data_train, data_validation, label_train)
# calculate prediction accuracy
score = NearestNeighbors.accuracy(predict, label_validation)
score = score * 100
print('Akurasi : %s' %score)