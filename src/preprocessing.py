import pandas as pd
import numpy as np


def detect_issues(daily: pd.DataFrame) -> dict:
    """
    Аналізує часовий ряд і повертає звіт про проблеми.
    daily — підготовлений DataFrame з колонками ds і y.
    """
    issues = {}

    # Дублікати дат
    duplicates = daily[daily['ds'].duplicated(keep=False)]
    if len(duplicates) > 0:
        issues['duplicates'] = {
            'count': daily['ds'].duplicated().sum(),
            'dates': duplicates['ds'].dt.strftime('%d.%m.%Y').tolist()
        }

    # Нульові значення
    zeros = daily[daily['y'] == 0]
    if len(zeros) > 0:
        issues['zeros'] = {
            'count': len(zeros),
            'dates': zeros['ds'].dt.strftime('%d.%m.%Y').tolist()
        }

    # Пропущені дати
    if len(daily) > 1:
        full_range = pd.date_range(
            start=daily['ds'].min(),
            end=daily['ds'].max(),
            freq='D'
        )
        missing_dates = full_range.difference(daily['ds'])
        if len(missing_dates) > 0:
            issues['missing_dates'] = {
                'count': len(missing_dates),
                'dates': [d.strftime('%d.%m.%Y') for d in missing_dates]
            }

    # Викиди — значення далі ніж 3 стандартних відхилення від середнього
    mean = daily['y'].mean()
    std = daily['y'].std()
    outliers = daily[
        (daily['y'] > mean + 3 * std) |
        (daily['y'] < mean - 3 * std)
    ]
    if len(outliers) > 0:
        issues['outliers'] = {
            'count': len(outliers),
            'dates': outliers['ds'].dt.strftime('%d.%m.%Y').tolist(),
            'values': outliers['y'].round(2).tolist()
        }

    return issues


def apply_fixes(daily: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """
    Застосовує виправлення на основі рішень користувача або автоматично.

    decisions — словник з рішеннями для кожної проблеми:
    {
        'duplicates': 'auto',       # завжди автоматично
        'zeros': 'closed',          # закриті дні — не враховувати
        'zeros': 'error',           # помилка — замінити середнім
        'outliers': 'special_day',  # особливий день — залишити
        'outliers': 'error',        # помилка — виправити автоматично
        'missing_dates': 'auto',    # завжди автоматично
    }
    """
    df = daily.copy()

    # Дублікати — завжди об'єднуємо сумою
    if 'duplicates' in decisions:
        df = df.groupby('ds')['y'].sum().reset_index()
        df = df.sort_values('ds').reset_index(drop=True)

    # Нулі
    if 'zeros' in decisions:
        if decisions['zeros'] == 'closed':
            # Закриті дні — замінюємо на NaN, Prophet пропустить
            df.loc[df['y'] == 0, 'y'] = np.nan
        elif decisions['zeros'] == 'error':
            # Помилка запису — замінюємо середнім сусідніх днів
            df['y'] = df['y'].replace(0, np.nan)
            df['y'] = df['y'].interpolate(method='linear')

    # Викиди
    if 'outliers' in decisions:
        if decisions['outliers'] == 'error':
            # Помилка — замінюємо медіаною
            median = df['y'].median()
            mean = df['y'].mean()
            std = df['y'].std()
            outlier_mask = (
                (df['y'] > mean + 3 * std) |
                (df['y'] < mean - 3 * std)
            )
            df.loc[outlier_mask, 'y'] = median
        # Якщо 'special_day' — нічого не робимо, залишаємо як є

    return df


def get_quality_score(issues: dict) -> tuple[int, str]:
    """
    Повертає оцінку якості даних від 0 до 100 і текстовий опис.
    """
    score = 100

    if 'duplicates' in issues:
        score -= issues['duplicates']['count'] * 5
    if 'zeros' in issues:
        score -= issues['zeros']['count'] * 3
    if 'missing_dates' in issues:
        score -= min(issues['missing_dates']['count'] * 2, 20)
    if 'outliers' in issues:
        score -= issues['outliers']['count'] * 10

    score = max(0, score)

    if score >= 80:
        label = "Висока якість даних"
    elif score >= 60:
        label = "Прийнятна якість даних"
    else:
        label = "Низька якість даних — рекомендується перевірити дані"

    return score, label