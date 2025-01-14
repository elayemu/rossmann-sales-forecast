import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib


def create_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(50, activation="relu", input_shape=input_shape, return_sequences=True),
            LSTM(50, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm():
    data = pd.read_csv("../data/train.csv", parse_dates=["Date"])
    data = data.sort_values("Date")

    # Prepare time series data
    sales = data["Sales"].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sales_scaled = scaler.fit_transform(sales.reshape(-1, 1))

    X, y = [], []
    for i in range(30, len(sales_scaled)):
        X.append(sales_scaled[i - 30 : i])
        y.append(sales_scaled[i])
    X, y = np.array(X), np.array(y)

    model = create_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, batch_size=32)

    model.save("../models/lstm_model.h5")
    joblib.dump(scaler, "../models/lstm_scaler.pkl")
