import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

st.title("AAPL Close Price Forecasting (Univariate LSTM)")

# Load cleaned dataset
df = pd.read_csv("stock_prices.csv")
close_series = df["Close"].values.reshape(-1, 1)

# Fit scaler on train portion
train_size = int(len(close_series) * 0.8)
scaler = MinMaxScaler()
scaler.fit(close_series[:train_size])

# Forecast input
window = 30
input_seq = st.text_area("Paste last 30 closing prices (comma-separated)")

# Load model
model = load_model("model.h5")

if st.button("Predict"):
    try:
        seq = np.array([float(x.strip()) for x in input_seq.split(",")])
        if len(seq) != window:
            st.error("Need exactly 30 values")
        else:
            seq = seq.reshape(window, 1)
            seq = scaler.transform(seq).reshape(1, window, 1)
            pred = model.predict(seq)
            st.write("Next Close Prediction:", round(float(scaler.inverse_transform(pred)[0][0]), 2))
    except:
        st.error("Invalid input format")

# Optional static evaluation preview
scaled_all = scaler.transform(close_series)
X, y = [], []
for i in range(len(scaled_all) - window):
    X.append(scaled_all[i:i+window])
    y.append(scaled_all[i+window])
X = np.array(X).reshape(len(X), window, 1)

pred_all = scaler.inverse_transform(model.predict(X))
actual_all = scaler.inverse_transform(np.array(y).reshape(-1,1))

plt.plot(actual_all, label="Actual")
plt.plot(pred_all, label="Predicted")
plt.legend()
st.pyplot(plt)

mae = mean_absolute_error(actual_all, pred_all)
rmse = math.sqrt(mean_squared_error(actual_all, pred_all))
st.write(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
