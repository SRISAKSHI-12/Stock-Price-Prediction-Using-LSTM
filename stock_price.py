import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM

import preprocessing

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(7)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("apple_share_price.csv", usecols=[1,2,3,4])

# Reverse dataset
dataset = dataset.iloc[::-1].reset_index(drop=True)

# -----------------------------
# Create Index
# -----------------------------
obs = np.arange(1, len(dataset)+1)

# -----------------------------
# Indicators
# -----------------------------
OHLC_avg = dataset.mean(axis=1)
HLC_avg = dataset[['High','Low','Close']].mean(axis=1)
close_val = dataset[['Close']]

# -----------------------------
# Plot Original Data
# -----------------------------
plt.figure(figsize=(12,6))

plt.plot(obs, OHLC_avg, 'r', label='OHLC Average')
plt.plot(obs, HLC_avg, 'b', label='HLC Average')
plt.plot(obs, close_val, 'g', label='Close Price')

plt.legend()
plt.title("Apple Stock Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# -----------------------------
# Scaling
# -----------------------------
OHLC_avg = np.reshape(OHLC_avg.values, (-1,1))

scaler = MinMaxScaler(feature_range=(0,1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# -----------------------------
# Train Test Split
# -----------------------------
train_size = int(len(OHLC_avg)*0.75)

train = OHLC_avg[:train_size]
test = OHLC_avg[train_size:]

# -----------------------------
# Prepare Dataset
# -----------------------------
step_size = 1

trainX, trainY = preprocessing.new_dataset(train, step_size)
testX, testY = preprocessing.new_dataset(test, step_size)

# -----------------------------
# Reshape
# -----------------------------
trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])
testX = testX.reshape(testX.shape[0],1,testX.shape[1])

# -----------------------------
# LSTM Model
# -----------------------------
model = Sequential()

model.add(
    LSTM(
        32,
        input_shape=(1,step_size),
        return_sequences=True
    )
)

model.add(LSTM(16))

model.add(Dense(1))

model.add(Activation("linear"))

# -----------------------------
# Compile
# -----------------------------
model.compile(
    loss="mean_squared_error",
    optimizer="adam"
)

# -----------------------------
# Train
# -----------------------------
model.fit(
    trainX,
    trainY,
    epochs=10,
    batch_size=1,
    verbose=1
)

# -----------------------------
# Prediction
# -----------------------------
trainPredict = model.predict(trainX, verbose=0)
testPredict = model.predict(testX, verbose=0)

# -----------------------------
# Inverse Transform
# -----------------------------
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

trainY = scaler.inverse_transform([trainY])
testY = scaler.inverse_transform([testY])

# -----------------------------
# RMSE
# -----------------------------
trainScore = math.sqrt(
    mean_squared_error(trainY[0],trainPredict[:,0])
)

testScore = math.sqrt(
    mean_squared_error(testY[0],testPredict[:,0])
)

print("\nTrain RMSE:",round(trainScore,2))
print("Test RMSE:",round(testScore,2))

# -----------------------------
# Plot Prediction
# -----------------------------
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size] = trainPredict

testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:] = np.nan

testPredictPlot[
len(trainPredict)+(step_size*2)+1:
len(OHLC_avg)-1
] = testPredict

OHLC_avg_original = scaler.inverse_transform(OHLC_avg)

plt.figure(figsize=(12,6))

plt.plot(
    OHLC_avg_original,
    label="Original",
    color="green"
)

plt.plot(
    trainPredictPlot,
    label="Training Prediction",
    color="red"
)

plt.plot(
    testPredictPlot,
    label="Testing Prediction",
    color="blue"
)

plt.title("Stock Price Prediction using LSTM")

plt.xlabel("Days")
plt.ylabel("Price")

plt.legend()

plt.show()

# -----------------------------
# Predict Next Day
# -----------------------------
last_val = testPredict[-1]

last_val_scaled = scaler.transform(
    last_val.reshape(-1,1)
)

next_val = model.predict(
    last_val_scaled.reshape(1,1,1),
    verbose=0
)

next_price = scaler.inverse_transform(next_val)

print("\nLast Day Value :", float(last_val[0]))
print("Next Day Prediction :", float(next_price[0][0]))