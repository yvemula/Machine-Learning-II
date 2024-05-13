import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def build_and_train_model(X_train, y_train, time_step=100):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    return model

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('time_series_data.csv')
    model = build_and_train_model(X_train, y_train)
