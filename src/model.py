from prophet import Prophet

def train_and_forecast(daily, periods=30):
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(daily)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast