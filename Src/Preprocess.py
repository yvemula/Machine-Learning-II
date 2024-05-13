import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, time_step=100):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display basic information
    print(data.info())
    print(data.describe())
    
    # Assume the time series data is in a column named 'value'
    values = data['value'].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))
    
    # Prepare the data for LSTM
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    X, y = create_dataset(scaled_values, time_step)
    
    # Split the data into training and test sets
    train_size = int(len(X) * 0.67)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    
    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('time_series_data.csv')
