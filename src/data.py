import pandas as pd


def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        try:
            pd.to_datetime(df[col], dayfirst=True, infer_datetime_format=True)
            return col
        except Exception:
            continue
    return None


def prepare_daily(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, infer_datetime_format=True)
    daily = df.groupby(date_col)[value_col].sum().reset_index()
    daily.columns = ['ds', 'y']
    daily = daily.sort_values('ds').reset_index(drop=True)
    return daily


def get_stats(daily: pd.DataFrame) -> dict:
    return {
        'days': len(daily),
        'start': daily['ds'].min().strftime('%d.%m.%Y'),
        'end': daily['ds'].max().strftime('%d.%m.%Y'),
        'mean': round(daily['y'].mean(), 2),
        'min': round(daily['y'].min(), 2),
        'max': round(daily['y'].max(), 2),
    }