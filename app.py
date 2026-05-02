from src.data import load_and_prepare
from src.model import train_and_forecast
from src.metrics import calculate_metrics
import matplotlib.pyplot as plt

# Завантаження даних
daily = load_and_prepare('data/coffee_sales.xlsx')
print(f"Завантажено {len(daily)} днів")
print(daily.head())

# Навчання і прогноз
model, forecast = train_and_forecast(daily, periods=30)
print("\nПрогноз готовий")

# Метрики
metrics = calculate_metrics(daily, forecast)
print(f"\nMAE:  {metrics['MAE']}")
print(f"RMSE: {metrics['RMSE']}")
print(f"MAPE: {metrics['MAPE']}%")

# Графік
fig = model.plot(forecast)
plt.title('Прогноз виручки на 30 днів')
plt.savefig('outputs/forecast_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено")