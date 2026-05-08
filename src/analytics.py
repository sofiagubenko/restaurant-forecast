import pandas as pd
import numpy as np


def get_general_stats(daily: pd.DataFrame, currency: str = '$') -> dict:
    """Загальні показники бізнесу за весь період."""
    total = daily['y'].sum()
    mean = daily['y'].mean()
    best_day = daily.loc[daily['y'].idxmax()]
    worst_day = daily.loc[daily['y'].idxmin()]

    # Тренд — порівнюємо першу і другу половину
    mid = len(daily) // 2
    first_half_mean = daily.iloc[:mid]['y'].mean()
    second_half_mean = daily.iloc[mid:]['y'].mean()
    trend_pct = ((second_half_mean - first_half_mean) / first_half_mean) * 100

    return {
        'total': round(total, 2),
        'mean': round(mean, 2),
        'best_day': {
            'date': best_day['ds'].strftime('%d.%m.%Y'),
            'value': round(best_day['y'], 2)
        },
        'worst_day': {
            'date': worst_day['ds'].strftime('%d.%m.%Y'),
            'value': round(worst_day['y'], 2)
        },
        'trend_pct': round(trend_pct, 1),
        'trend_direction': 'зростання' if trend_pct > 0 else 'спадання'
    }


def get_weekday_stats(daily: pd.DataFrame) -> dict | None:
    """Аналітика по днях тижня. None якщо даних менше 14 днів."""
    if len(daily) < 14:
        return None

    days_ua = ['Понеділок', 'Вівторок', 'Середа',
               'Четвер', 'П\'ятниця', 'Субота', 'Неділя']

    df = daily.copy()
    df['weekday'] = df['ds'].dt.dayofweek
    weekday_avg = df.groupby('weekday')['y'].mean().round(2)

    best_idx = weekday_avg.idxmax()
    worst_idx = weekday_avg.idxmin()

    return {
        'averages': {days_ua[i]: weekday_avg.get(i, 0) for i in range(7)},
        'best': {'name': days_ua[best_idx], 'value': weekday_avg[best_idx]},
        'worst': {'name': days_ua[worst_idx], 'value': weekday_avg[worst_idx]}
    }


def get_monthly_stats(daily: pd.DataFrame) -> dict | None:
    """Аналітика по місяцях. None якщо даних менше 60 днів."""
    if len(daily) < 60:
        return None

    months_ua = {
        1: 'Січень', 2: 'Лютий', 3: 'Березень', 4: 'Квітень',
        5: 'Травень', 6: 'Червень', 7: 'Липень', 8: 'Серпень',
        9: 'Вересень', 10: 'Жовтень', 11: 'Листопад', 12: 'Грудень'
    }

    df = daily.copy()
    df['month'] = df['ds'].dt.month
    monthly_avg = df.groupby('month')['y'].mean().round(2)

    best_idx = monthly_avg.idxmax()
    worst_idx = monthly_avg.idxmin()

    return {
        'averages': {months_ua[i]: monthly_avg.get(i, 0)
                     for i in monthly_avg.index},
        'best': {'name': months_ua[best_idx], 'value': monthly_avg[best_idx]},
        'worst': {'name': months_ua[worst_idx], 'value': monthly_avg[worst_idx]}
    }


def get_comparison_stats(daily: pd.DataFrame) -> dict | None:
    """Порівняння першої і другої половини періоду. None якщо менше 30 днів."""
    if len(daily) < 30:
        return None

    mid = len(daily) // 2
    first = daily.iloc[:mid]
    second = daily.iloc[mid:]

    first_mean = first['y'].mean()
    second_mean = second['y'].mean()
    change_pct = ((second_mean - first_mean) / first_mean) * 100

    # Вихідні vs будні
    df = daily.copy()
    df['weekday'] = df['ds'].dt.dayofweek
    weekends = df[df['weekday'].isin([5, 6])]['y'].mean()
    weekdays = df[~df['weekday'].isin([5, 6])]['y'].mean()
    weekend_boost = ((weekends - weekdays) / weekdays) * 100

    return {
        'first_period': {
            'start': first['ds'].min().strftime('%d.%m.%Y'),
            'end': first['ds'].max().strftime('%d.%m.%Y'),
            'mean': round(first_mean, 2)
        },
        'second_period': {
            'start': second['ds'].min().strftime('%d.%m.%Y'),
            'end': second['ds'].max().strftime('%d.%m.%Y'),
            'mean': round(second_mean, 2)
        },
        'change_pct': round(change_pct, 1),
        'weekend_boost_pct': round(weekend_boost, 1)
    }


def generate_insights(daily: pd.DataFrame, currency: str = '$') -> list[str]:
    """
    Генерує список текстових висновків для власника.
    Кожен елемент — окреме речення.
    """
    insights = []

    general = get_general_stats(daily, currency)
    weekday = get_weekday_stats(daily)
    comparison = get_comparison_stats(daily)

    # Тренд
    direction = "виріс" if general['trend_pct'] > 0 else "впав"
    insights.append(
        f"За досліджуваний період бізнес {direction} на "
        f"{abs(general['trend_pct'])}%."
    )

    # Найкращий день тижня
    if weekday:
        insights.append(
            f"Найкращий день тижня — {weekday['best']['name']} "
            f"(середня виручка {currency}{weekday['best']['value']:.0f})."
        )
        insights.append(
            f"Найслабший день тижня — {weekday['worst']['name']} "
            f"(середня виручка {currency}{weekday['worst']['value']:.0f})."
        )

    # Вихідні vs будні
    if comparison:
        if comparison['weekend_boost_pct'] > 0:
            insights.append(
                f"Вихідні приносять на {comparison['weekend_boost_pct']}% "
                f"більше виручки ніж будні."
            )
        else:
            insights.append(
                f"Будні приносять на {abs(comparison['weekend_boost_pct'])}% "
                f"більше виручки ніж вихідні."
            )

    # Найкращий конкретний день
    insights.append(
        f"Найкращий день за весь період — {general['best_day']['date']} "
        f"({currency}{general['best_day']['value']:.0f})."
    )

    return insights