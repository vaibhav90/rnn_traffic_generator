import numpy as np
import csv
import s2sphere
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import itertools
import gmplot
from keras.models import load_model


f = open('xyz.csv', 'rU')
gps_logs = csv.reader(f)

raw_logs = []

for row in gps_logs:
    raw_logs.append(row)

logs = np.array(raw_logs)

time = logs[:,2]
lat = logs[:,0]
lng = logs[:,1]

r = s2sphere.RegionCoverer()

'''
grids = []
print float(lat[0]), float(lng[0])
p = s2sphere.LatLng.from_degrees(float(lat[0]), float(lng[0]))
c = s2sphere.CellId.from_lat_lng(p)
cellid = c.id()
print cellid
ll = s2sphere.CellId(cellid).to_lat_lng()
print ll
'''

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

dt = []

for i in range(0, len(lat)):
    p = s2sphere.LatLng.from_degrees(float(lat[i]), float(lng[i]))
    c = s2sphere.CellId.from_lat_lng(p)
    dt.append(c.id())

dataset = []
for row in dt:
    dataset.append(row)

df = pd.DataFrame(dataset)
dataset = df.values
dataset = dataset.astype('float64')
dataset = dataset[0:1000]


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.99)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=2)


#model.save('traffic_gen01.h5')  # creates a HDF5 file 'my_model.h5'

# serialize model to JSON
model_json = model.to_json()
with open("traffic_gen01.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("traffic_gen.h5")
print("Saved model to disk")


'''
print 'training done!'


predict = model.predict(testX)

trajectory = []

for i in range(0, 1000):

    p = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))

    predict = model.predict(p)

    trajectory.append(scaler.inverse_transform(predict))


s_c_id = list(itertools.chain(*trajectory))


cellId = []

for i in range(0, len(s_c_id)):
    cellId.append(s_c_id[i][0])

cellId = map(int, cellId)


map_lat = []
map_lng = []
for i in range(0, len(s_c_id)):
    ll = str(s2sphere.CellId(cellId[i]).to_lat_lng())
    latlng = ll.split(',', 1)
    lat = latlng[0].split(':', 1)
    map_lat.append(float(lat[1]))
    map_lng.append(float(latlng[1]))


#print map_lat
#print map_lng

plt.plot(map_lat, map_lng)
plt.show()


gmap = gmplot.GoogleMapPlotter(46.519962, 6.633597, 16)

gmap.plot(map_lat, map_lng, '#000000', edge_width=20)

#gmap.scatter(map_lat, map_lng, '#000000', edge_width=20)

gmap.draw("map001.html")


'''
