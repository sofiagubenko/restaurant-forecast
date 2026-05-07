import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.data import load_file, detect_date_column, prepare_daily, get_stats
from src.model import train_and_forecast, split_train_test
from src.metrics import calculate_metrics, generate_summary

# ---- Налаштування сторінки ----
st.set_page_config(
    page_title="Прогнозування попиту ресторану",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Прогнозування попиту на послуги ресторану")
st.markdown("Завантажте файл з історичними даними продажів — система побудує прогноз на обраний період.")

# ---- Бокова панель ----
with st.sidebar:
    st.header("⚙️ Налаштування")

    uploaded_file = st.file_uploader(
        "Завантажте файл даних",
        type=["csv", "xlsx", "xls"],
        help="CSV або Excel файл з даними продажів"
    )

    st.divider()

    periods = st.selectbox(
        "Горизонт прогнозування",
        options=[7, 14, 30, 60],
        index=2,
        format_func=lambda x: f"{x} днів"
    )

    country = st.selectbox(
        "Країна (для врахування свят)",
        options=["Без свят", "US", "UA", "DE", "PL"],
        index=0
    )
    country_code = None if country == "Без свят" else country

    currency = st.text_input("Символ валюти", value="$")

    run_button = st.button("🚀 Запустити прогноз", type="primary", use_container_width=True)

# ---- Основна область ----
if uploaded_file is None:
    st.info("👈 Завантажте файл з даними у боковій панелі щоб почати.")
    st.stop()

# Завантаження файлу
df = load_file(uploaded_file)

# Вибір колонок
st.subheader("📋 Налаштування колонок")
col1, col2 = st.columns(2)

with col1:
    detected_date = detect_date_column(df)
    date_col = st.selectbox(
        "Колонка з датою",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(detected_date) if detected_date else 0
    )

with col2:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    value_col = st.selectbox(
        "Колонка зі значенням",
        options=numeric_cols
    )

# Попередній перегляд
with st.expander("👀 Попередній перегляд даних", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# Підготовка даних
daily = prepare_daily(df, date_col, value_col)
stats = get_stats(daily)

# Статистика
st.subheader("📊 Статистика датасету")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Кількість днів", stats['days'])
c2.metric("Початок", stats['start'])
c3.metric("Кінець", stats['end'])
c4.metric(f"Середня виручка", f"{currency}{stats['mean']}")
c5.metric("Макс. виручка", f"{currency}{stats['max']}")

# ---- Запуск прогнозу ----
if run_button:
    with st.spinner("Навчання моделі та побудова прогнозу..."):

        # Розбивка на train/test
        train, test = split_train_test(daily, test_days=min(30, len(daily) // 5))

        # Навчання на train, оцінка на test
        _, forecast_test = train_and_forecast(train, periods=len(test), country=country_code)
        metrics = calculate_metrics(test, forecast_test)

        # Навчання на всіх даних для фінального прогнозу
        model, forecast = train_and_forecast(daily, periods=periods, country=country_code)

    st.success("Прогноз готовий!")

    # ---- Метрики ----
    st.subheader("📈 Якість моделі")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{currency}{metrics['MAE']}", help="Середня абсолютна похибка")
    m2.metric("RMSE", f"{currency}{metrics['RMSE']}", help="Середньоквадратична похибка")
    m3.metric("MAPE", f"{metrics['MAPE']}%", help="Середня відсоткова похибка")

    summary = generate_summary(metrics, currency)
    st.info(f"💡 {summary}")

    # ---- Графік прогнозу ----
    st.subheader("📉 Прогноз виручки")

    fig = go.Figure()

    # Фактичні дані
    fig.add_trace(go.Scatter(
        x=daily['ds'],
        y=daily['y'],
        mode='markers',
        name='Фактична виручка',
        marker=dict(color='#2E86AB', size=5, opacity=0.7)
    ))

    # Довірчий інтервал
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Довірчий інтервал',
        showlegend=True
    ))

    # Лінія прогнозу
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Прогноз',
        line=dict(color='#FF6B35', width=2)
    ))

    # Вертикальна лінія — початок прогнозу
    last_date = daily['ds'].max().strftime('%Y-%m-%d')
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", dash="dash")
    )
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Початок прогнозу",
        showarrow=False,
        yshift=10,
        font=dict(color="gray")
    )

    fig.update_layout(
        xaxis_title="Дата",
        yaxis_title=f"Виручка ({currency})",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Графік компонентів ----
    st.subheader("🔍 Компоненти моделі")

    col_trend, col_weekly = st.columns(2)

    with col_trend:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            line=dict(color='#2E86AB', width=2),
            name='Тренд'
        ))
        fig_trend.update_layout(
            title="Тренд",
            xaxis_title="Дата",
            yaxis_title=f"Виручка ({currency})",
            height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_weekly:
        days_ua = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд']
        weekly_cols = [c for c in forecast.columns if 'weekly' in c]
        if weekly_cols:
            weekly_data = forecast[['ds'] + weekly_cols].copy()
            weekly_data['weekday'] = weekly_data['ds'].dt.dayofweek
            weekly_avg = weekly_data.groupby('weekday')[weekly_cols[0]].mean()

            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Bar(
                x=days_ua,
                y=weekly_avg.values,
                marker_color='#FF6B35',
                name='Сезонність'
            ))
            fig_weekly.update_layout(
                title="Тижнева сезонність",
                xaxis_title="День тижня",
                yaxis_title="Відхилення від тренду",
                height=300
            )
            st.plotly_chart(fig_weekly, use_container_width=True)

    # ---- Таблиця прогнозу ----
    st.subheader("📅 Деталі прогнозу")
    forecast_display = forecast[forecast['ds'] > daily['ds'].max()][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].copy()
    forecast_display.columns = ['Дата', 'Прогноз', 'Мінімум', 'Максимум']
    forecast_display['Дата'] = forecast_display['Дата'].dt.strftime('%d.%m.%Y')
    for col in ['Прогноз', 'Мінімум', 'Максимум']:
        forecast_display[col] = forecast_display[col].apply(lambda x: f"{currency}{x:.2f}")
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)

    # ---- Експорт ----
    st.subheader("💾 Експорт результатів")
    e1, e2 = st.columns(2)

    with e1:
        csv = forecast_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Завантажити прогноз (CSV)",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

    with e2:
        fig_bytes = fig.to_image(format="png", width=1200, height=500, scale=2)
        st.download_button(
            label="⬇️ Завантажити графік (PNG)",
            data=fig_bytes,
            file_name="forecast_plot.png",
            mime="image/png",
            use_container_width=True
        )