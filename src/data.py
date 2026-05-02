import pandas as pd

def load_and_prepare(filepath):
    df = pd.read_excel(filepath)  # замість read_csv
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['revenue'] = df['transaction_qty'] * df['unit_price']
    daily = df.groupby('transaction_date')['revenue'].sum().reset_index()
    daily.columns = ['ds', 'y']
    return daily