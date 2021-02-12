import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend

dataframe = pd.read_excel('D:\dekstop\Flask_app\KOE_database.xlsx')

# fix random seed for reproducibility
np.random.seed(7)

dataframe.drop('time', inplace=True, axis=1)

dataset = dataframe.values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_x = dataset[0:train_size, 0:-1]
train_y = dataset[0:train_size, -1]
test_x = dataset[0:test_size, 0:-1]
test_y = dataset[0:test_size, -1]

from sklearn import preprocessing as pp

# Prepocess data - Scaling X data (Min-Max)
min_max_scaler = pp.MinMaxScaler()
train_x_scale = min_max_scaler.fit_transform(train_x)
test_x_scale = min_max_scaler.transform(test_x)

# Preprocess data - Scaling Y data (Normalize)
train_y_norm = train_y / max(train_y)
test_y_norm = test_y / max(train_y)

# Create Neural Network Model
model = Sequential()                       # Initialization of model
model.add(Input(shape=(4,)))               # Input layer
model.add(Dense(2,activation='relu'))      # 1st Hidden Layer with 2 nodes and Activation Function = Sigmoid
model.add(Dense(1))                        # Output layer with 1 node and Activation Function = Linear [y = x]

model.compile(optimizer='adam',
              loss='mse')

# Train Neural Network Model
history = model.fit(train_x_scale, train_y_norm,
          batch_size=32, epochs=200, validation_data=(test_x, test_y))


print(model.summary())\

import keras
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.show()

# make predictions
trainPredict = model.predict(train_x_scale)
testPredict = model.predict(test_x_scale)

from sklearn.metrics import r2_score

# Evaluate the model
loss = model.evaluate(test_x_scale, test_y_norm)
print("loss (MSE):",loss)

# R2 coefficient
r2 = r2_score(test_y_norm, testPredict)
print("R2:",r2)

import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_y_norm, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore*100))
testScore = math.sqrt(mean_squared_error(test_y_norm, testPredict))
print('Test Score: %.2f RMSE' % (testScore*100))

#calculate mean absolute percent error
trainMAPE = mean_absolute_error(train_y, trainPredict)
print('train MAPE: %.2f MAPE' % (trainMAPE*100))
testMAPE = mean_absolute_error(test_y, testPredict)
print('test MAPE: %.2f MAPE' % (testMAPE*100))


# Get something which has as many features as dataset
testPredict_extended = np.zeros((len(testPredict),5))
# Put the predictions there
testPredict_extended[:,4] = testPredict[:,0]
# Inverse transform it and select the 5rd column.
testPredict = scaler.inverse_transform(testPredict_extended)[:,4]
print('testPredict',testPredict)

model.save("prediction.h5")