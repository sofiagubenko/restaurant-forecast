import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(daily, forecast):
    merged = daily.merge(
        forecast[['ds', 'yhat']], on='ds'
    )
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    mape = (abs((merged['y'] - merged['yhat']) / merged['y']).mean()) * 100
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}