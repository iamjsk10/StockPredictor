import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from keras.src.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#Load Data
company = 'AAPL'
start = '2020-01-01'
end = '2024-01-01'

data = yf.download(company, 'yahoo', start,end)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

predictions_days = 60

x_train = []
y_train = []

for x in range(predictions_days, len(scaled_data)):
    x_train.append(scaled_data[x-predictions_days:x,0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1,1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # prediciton of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Load test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime(2021,1,1)

test_data = yf.download(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - predictions_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make prediction on test Data

x_test = []

for x in range(predictions_days, len(model_inputs)):
    x_test.append(model_inputs[x-predictions_days:x,0])

x_test = np.array(x_test)
x_test = np.array(x_test.shape[0],( x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#Plot test predictions
plt.plot(actual_prices, label=f"Actual {company} prices",color = "black")
plt.plot(predicted_prices, label=f"Predicted {company}prices",color = "blue")
plt.title(f"{company} Share Price")

plt.legend()
plt.show()




