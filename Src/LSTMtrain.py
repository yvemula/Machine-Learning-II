import tensorflow as tf
import tensorflow
import keras
from tensorflow.python.keras.layers import Input, Dense, LSTM
from tensorflow.python.keras.models import Sequential
from keras import layers
from keras import models


def build_and_train_model(X_train, y_train, epochs=1, batch_size=1):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    return model

if __name__ == "__main__":
    from Preprocess import load_and_preprocess_data
    
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('time_series_data.csv')
    model = build_and_train_model(X_train, y_train)
