import pandas as pd
from prophet import Prophet


def train_and_forecast(daily: pd.DataFrame, periods: int = 30, country: str = None):
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8
    )

    if country:
        model.add_country_holidays(country_name=country)

    model.fit(daily)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return model, forecast


def split_train_test(daily: pd.DataFrame, test_days: int = 30):
    """Розбиває датасет на навчальну і тестову вибірки."""
    train = daily.iloc[:-test_days]
    test = daily.iloc[-test_days:]
    return train, test