import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

def evaluate_and_visualize(model, X_train, X_test, y_train, y_test, scaler, scaled_values, time_step=100):
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform to get actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_actual = scaler.inverse_transform([y_train])
    y_test_actual = scaler.inverse_transform([y_test])
    
    # Evaluate the model
    train_rmse = math.sqrt(mean_squared_error(y_train_actual[0], train_predict[:, 0]))
    test_rmse = math.sqrt(mean_squared_error(y_test_actual[0], test_predict[:, 0]))
    
    print('Train RMSE: ', train_rmse)
    print('Test RMSE: ', test_rmse)
    
    # Plot the results
    train_plot = np.empty_like(scaled_values)
    train_plot[:, :] = np.nan
    train_plot[time_step:len(train_predict) + time_step, :] = train_predict
    
    test_plot = np.empty_like(scaled_values)
    test_plot[:, :] = np.nan
    test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_values) - 1, :] = test_predict
    
    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(scaled_values), label='Original Data')
    plt.plot(train_plot, label='Train Predictions')
    plt.plot(test_plot, label='Test Predictions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from Preprocess import load_and_preprocess_data
    from LSTMtrain import build_and_train_model
    
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('time_series_data.csv')
    model = build_and_train_model(X_train, y_train)
    evaluate_and_visualize(model, X_train, X_test, y_train, y_test, scaler, X_train)  # Pass scaled_values argument
