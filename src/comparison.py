import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def baseline_forecast(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Baseline модель — ковзне середнє за останні 7 днів.
    Повертає DataFrame з колонками ds і yhat.
    """
    window = 7
    last_values = train['y'].tail(window).values
    mean_val = last_values.mean()

    future_dates = pd.date_range(
        start=train['ds'].max() + pd.Timedelta(days=1),
        periods=periods,
        freq='D'
    )

    return pd.DataFrame({
        'ds': future_dates,
        'yhat': [mean_val] * periods
    })


def arima_forecast(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    ARIMA модель з автоматичним підбором порядку.
    Повертає DataFrame з колонками ds і yhat.
    """
    y = train['y'].values

    # Пробуємо кілька конфігурацій і беремо найкращу за AIC
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(y, order=(p, d, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    # Навчаємо з найкращими параметрами
    model = ARIMA(y, order=best_order)
    result = model.fit()
    forecast_values = result.forecast(steps=periods)

    future_dates = pd.date_range(
        start=train['ds'].max() + pd.Timedelta(days=1),
        periods=periods,
        freq='D'
    )

    return pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values
    }), best_order


def prophet_forecast(train: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Prophet модель.
    Повертає DataFrame з колонками ds і yhat.
    """
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8
    )
    model.fit(train)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]


def calculate_metrics(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """
    Розраховує MAE, RMSE, MAPE для пари actual/predicted.
    """
    merged = actual.merge(predicted[['ds', 'yhat']], on='ds')
    if len(merged) == 0:
        return {'MAE': None, 'RMSE': None, 'MAPE': None}

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    mape = (abs((merged['y'] - merged['yhat']) / merged['y']).mean()) * 100

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2)
    }


def compare_models(daily: pd.DataFrame, test_days: int = 30) -> dict:
    """
    Порівнює три моделі на тестовій вибірці.
    Повертає словник з метриками для кожної моделі.
    """
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()

    results = {}

    # Baseline
    baseline_pred = baseline_forecast(train, test_days)
    results['Baseline (ковзне середнє)'] = calculate_metrics(test, baseline_pred)

    # ARIMA
    try:
        arima_pred, best_order = arima_forecast(train, test_days)
        results['ARIMA'] = calculate_metrics(test, arima_pred)
        results['ARIMA']['order'] = str(best_order)
    except Exception as e:
        results['ARIMA'] = {'MAE': None, 'RMSE': None, 'MAPE': None, 'error': str(e)}

    # Prophet
    prophet_pred = prophet_forecast(train, test_days)
    results['Prophet'] = calculate_metrics(test, prophet_pred)

    return results