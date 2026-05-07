import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(daily: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    merged = daily.merge(forecast[['ds', 'yhat']], on='ds')

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    mape = (abs((merged['y'] - merged['yhat']) / merged['y']).mean()) * 100

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2)
    }


def generate_summary(metrics: dict, currency: str = '$') -> str:
    """Генерує текстове резюме результатів прогнозування."""
    mape = metrics['MAPE']
    mae = metrics['MAE']

    if mape < 10:
        quality = "відмінну"
        recommendation = "Прогноз можна використовувати для операційного планування."
    elif mape < 20:
        quality = "прийнятну"
        recommendation = "Прогноз підходить для загального планування закупівель."
    else:
        quality = "низьку"
        recommendation = "Рекомендується зібрати більше історичних даних для покращення точності."

    return (
        f"Модель демонструє {quality} точність прогнозування. "
        f"Середня похибка складає {currency}{mae} на день ({mape:.1f}% від фактичної виручки). "
        f"{recommendation}"
    )