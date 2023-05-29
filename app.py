start = '2010-01-01'
end = '2019-12-30'

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(user_input, start='2010-01-01', end='2019-12-30')

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = go.Figure(data=go.Scatter(x=df.index, y=df.Close, mode='lines'))
st.plotly_chart(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100-day Moving Average'))
fig.add_trace(go.Scatter(x=df.index, y=df.Close, mode='lines', name='Closing Price'))
st.plotly_chart(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100-day Moving Average'))
fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200-day Moving Average'))
fig.add_trace(go.Scatter(x=df.index, y=df.Close, mode='lines', name='Closing Price'))
st.plotly_chart(fig)

data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='Original Price'))
fig2.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_predicted.flatten(), mode='lines', name='Predicted Price'))
st.plotly_chart(fig2)
