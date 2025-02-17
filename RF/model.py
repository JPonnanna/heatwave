import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def build_model(n_estimators=100, n_features=1):
    """Build and return a RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    return model

def train_model(model, X_train, Y_train, X_test, Y_test):
    """Train the model and evaluate it."""
    model.fit(X_train, Y_train)  # Train the RandomForest model
    
    Y_predict = model.predict(X_test)  # Get predictions from the model
    
    # Evaluate performance using RMSE and MAE
    rmse = sqrt(mean_squared_error(Y_predict, Y_test))
    mae = mean_absolute_error(Y_predict, Y_test)
    
    return rmse, mae, Y_predict, Y_test

def evaluate_model(model, scaler, X_test, Y_test):
    """Evaluate the model on the test set and inverse transform the results."""
    Y_predict = model.predict(X_test)
    
    # Since we don't scale predictions, no need for inverse transformation here.
    Y_predict = np.array(Y_predict)
    
    # Inverse transform the actual values (Y_test)
    temp = np.zeros((Y_test.shape[0], 1))
    for i in range(Y_test.shape[0]):
        temp[i][0] = Y_test[i]
    Y_test = scaler.inverse_transform(temp)  # Rescale back to original values
    Y_test = Y_test[:, 0]
    
    # Calculate RMSE and MAE
    rmse = sqrt(mean_squared_error(Y_predict, Y_test))
    mae = mean_absolute_error(Y_predict, Y_test)
    
    return rmse, mae, Y_predict, Y_test

# Sample code for usage:
# scaler = MinMaxScaler(feature_range=(0, 1))  # Example scaler, if needed
# X_train, Y_train, X_test, Y_test = prepare_data()  # Replace with actual data preparation
# model = build_model()
# rmse, mae, Y_pred, Y_test = train_model(model, X_train, Y_train, X_test, Y_test)
# print(f"RMSE: {rmse}, MAE: {mae}")
