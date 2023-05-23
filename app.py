import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

start = '2010-01-01'
end = '2019-12-30'
# start = datetime(2022, 1, 1)
# end = datetime(2022, 12, 31)


st.title('Stock Trend Prediction')

# user_input = st.text_input("Enter Stock Ticker", "AAPL")
# df = data.DataReader(user_input, 'yahoo', start, end)

user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(user_input, start='2010-01-01', end='2019-12-30')

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

















st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)




# data_training = pd.DataFrame (df['Close'][0: int(len(df)*0.70)])
# data_testing = pd.DataFrame(df['Close'][int (len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# data_training_array = scaler.fit_transform(data_training)

 





# model = load_model('keras_model.h5')



# past_100_days = data_training.tail(100)
# data_testing = pd.DataFrame()
# # final_df = past_100_days.append(data_testing, ignore_index=True)

# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)


# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#   x_test.append(input_data[i-100: i])
#   y_test.append(input_data[i, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)
# y_predicted = model.predict(x_test)
# scaler = scaler.scale_

# scale_factor = 1/scaler[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor




# st.subheader('Predictins vs original')
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label='Original Price')
# plt.plot(y_predicted, 'r', label ='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)









data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
# data_training_scaled = scaler.fit_transform(data_training)

data_training_array = scaler.fit_transform(data_training)




model = load_model('keras_model.h5')

# past_100_days = data_training_scaled[-100:]
past_100_days = data_training.tail(100)


final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

# for i in range(100, len(data_testing)):
#     x_test.append(data_training_scaled[i-100:i])
#     y_test.append(data_testing.iloc[i, 0])

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])


x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)


scaler = scaler.scale_
# y_predicted = y_predicted.reshape(-1)  # Flatten the predictions

# Scale back the values
# scale_factor = 1 / scaler.scale_[0]
# y_test = y_test * scale_factor
# y_predicted = y_predicted * scale_factor


scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)





























# Assuming you have defined `past_100_days`, `data_testing`, and `final_df` as before



# Assuming you have defined `past_100_days`, `data_testing`, and `final_df` as before

# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# # Rest of the code remains the same
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100: i])
#     y_test.append(input_data[i, 0])

# x_test, y_test = tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

# # Enable eager execution
# tf.config.run_functions_eagerly(True)

# y_predicted = model.predict(x_test)

# # Disable eager execution
# tf.config.run_functions_eagerly(False)

# scaler = scaler.scale_

# scale_factor = 1 / scaler[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor

# st.subheader('Predictions vs original')
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label='Original Price')
# plt.plot(y_predicted, 'r', label ='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)
