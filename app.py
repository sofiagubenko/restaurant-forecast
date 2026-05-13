import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.data import load_file, detect_date_column, prepare_daily, get_stats
from src.model import train_and_forecast, split_train_test
from src.metrics import calculate_metrics, generate_summary
from src.preprocessing import detect_issues, apply_fixes, get_quality_score
from src.analytics import (
    get_general_stats, get_weekday_stats,
    get_monthly_stats, get_comparison_stats, generate_insights
)

# ---- Налаштування сторінки ----
st.set_page_config(
    page_title="Прогнозування попиту ресторану",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Прогнозування попиту на послуги ресторану")
st.markdown("Завантажте файл з історичними даними продажів — система проаналізує бізнес і побудує прогноз.")

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

# ---- Захист від порожнього файлу ----
if uploaded_file is None:
    st.info("👈 Завантажте файл з даними у боковій панелі щоб почати.")
    st.stop()


# Завантаження файлу
try:
    df = load_file(uploaded_file)
except Exception as e:
    st.error(f"❌ Не вдалося відкрити файл. Переконайтесь що файл не пошкоджений і має формат CSV або Excel.")
    st.stop()

# Перевірка що файл не порожній
if df.empty:
    st.error("❌ Файл порожній — немає даних для аналізу.")
    st.stop()

if len(df) < 2:
    st.error("❌ Файл містить менше 2 рядків — недостатньо даних.")
    st.stop()

# Вибір колонок
st.subheader("📋 Налаштування колонок")
col1, col2 = st.columns(2)

with col1:
    detected_date = detect_date_column(df)
    if detected_date is None:
        st.warning("⚠️ Не вдалося автоматично визначити колонку з датою. Оберіть її вручну.")
    date_col = st.selectbox(
        "Колонка з датою",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(detected_date) if detected_date else 0
    )

with col2:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("❌ Файл не містить числових колонок — неможливо визначити виручку.")
        st.stop()

    calc_revenue = st.checkbox(
        "Розрахувати виручку як добуток двох колонок",
        help="Увімкніть якщо у вас є окремі колонки кількості та ціни"
    )

if calc_revenue:
    col3, col4 = st.columns(2)
    with col3:
        qty_col = st.selectbox("Колонка кількості", options=numeric_cols)
    with col4:
        price_col = st.selectbox(
            "Колонка ціни",
            options=numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0
        )
    if qty_col == price_col:
        st.error("❌ Колонка кількості і колонка ціни не можуть бути однаковими.")
        st.stop()
    df['revenue'] = df[qty_col] * df[price_col]
    value_col = 'revenue'
else:
    value_col = st.selectbox(
        "Колонка зі значенням",
        options=numeric_cols
    )

with st.expander("👀 Попередній перегляд даних", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ---- Фільтрація по категорії ----
categorical_cols = [
    col for col in df.columns
    if col not in [date_col, value_col]
    and (df[col].dtype == 'object' or (df[col].dtype in ['int64', 'int32'] and df[col].nunique() <= 50))
    and df[col].nunique() <= 50
    and df[col].nunique() >= 2
]

if categorical_cols:
    st.subheader("🔍 Фільтрація даних")
    filter_col = st.selectbox(
        "Фільтрувати по колонці (необов'язково)",
        options=["Без фільтру"] + categorical_cols
    )

    if filter_col != "Без фільтру":
        unique_values = sorted(df[filter_col].dropna().unique().tolist())
        selected_values = st.multiselect(
            f"Оберіть значення ({filter_col})",
            options=unique_values,
            default=unique_values
        )
        if selected_values:
            df = df[df[filter_col].isin(selected_values)].copy()
            st.info(f"🔍 Відфільтровано: {len(df)} рядків з {filter_col} = {', '.join(map(str, selected_values))}")
        else:
            st.warning("⚠️ Не обрано жодного значення — показуються всі дані.")

# Підготовка даних
try:
    daily_raw = prepare_daily(df, date_col, value_col)
except Exception as e:
    st.error(f"❌ Помилка при підготовці даних: не вдалося обробити колонку дати. Спробуйте вибрати іншу колонку.")
    st.stop()

# Перевірка після підготовки
if len(daily_raw) < 7:
    st.error(f"❌ Після агрегації залишилось лише {len(daily_raw)} днів — недостатньо для аналізу. Потрібно мінімум 7 днів.")
    st.stop()

if daily_raw['y'].sum() == 0:
    st.error("❌ Всі значення виручки дорівнюють нулю — перевірте правильність вибраних колонок.")
    st.stop()

if daily_raw['y'].isna().all():
    st.error("❌ Колонка значень містить лише порожні дані — перевірте правильність вибраних колонок.")
    st.stop()

# ---- Вкладки ----
tab1, tab2, tab3 = st.tabs([
    "📋 Якість даних",
    "📊 Аналітика бізнесу",
    "🔮 Прогноз"
])

# ========== ВКЛАДКА 1 — ЯКІСТЬ ДАНИХ ==========
with tab1:
    st.subheader("Перевірка якості даних")

    issues = detect_issues(daily_raw)
    score, label = get_quality_score(issues)

    # Оцінка якості
    col_score, col_label = st.columns([1, 3])
    with col_score:
        color = "green" if score >= 80 else "orange" if score >= 60 else "red"
        st.metric("Оцінка якості", f"{score}/100")
    with col_label:
        st.info(f"**{label}**")

    if not issues:
        st.success("✅ Проблем не знайдено — дані готові до аналізу.")
        daily = daily_raw
    else:
        # Показуємо знайдені проблеми
        st.warning(f"⚠️ Знайдено {len(issues)} тип(и) проблем:")

        for key, val in issues.items():
            if key == 'duplicates':
                st.write(f"**Дублікати дат:** {val['count']} шт.")
            elif key == 'zeros':
                st.write(f"**Нульові значення:** {val['count']} днів — {', '.join(val['dates'][:5])}")
            elif key == 'missing_dates':
                st.write(f"**Пропущені дати:** {val['count']} днів")
            elif key == 'outliers':
                details = [f"{d} (${v})" for d, v in
                          zip(val['dates'][:3], val['values'][:3])]
                st.write(f"**Викиди:** {val['count']} днів — {', '.join(details)}")

        # Автоматичні рішення за замовчуванням
        decisions = {}
        if 'duplicates' in issues:
            decisions['duplicates'] = 'auto'
        if 'zeros' in issues:
            decisions['zeros'] = 'closed'
        if 'outliers' in issues:
            decisions['outliers'] = 'error'

        # Розширені налаштування
        with st.expander("🔧 Розширені налаштування"):
            st.caption("Змініть як система має обробляти кожну проблему.")

            if 'zeros' in issues:
                zero_choice = st.radio(
                    f"Нульові значення ({issues['zeros']['count']} днів):",
                    options=["closed", "error"],
                    format_func=lambda x: (
                        "Ці дні були закриті — не враховувати в прогнозі"
                        if x == "closed"
                        else "Це помилка запису — замінити середнім значенням"
                    ),
                    key="zero_choice"
                )
                decisions['zeros'] = zero_choice

            if 'outliers' in issues:
                outlier_choice = st.radio(
                    f"Викиди ({issues['outliers']['count']} днів):",
                    options=["error", "special_day"],
                    format_func=lambda x: (
                        "Це помилка — виправити автоматично"
                        if x == "error"
                        else "Це був особливий день (акція, подія) — врахувати в прогнозі"
                    ),
                    key="outlier_choice"
                )
                decisions['outliers'] = outlier_choice

        # Застосовуємо виправлення
        daily = apply_fixes(daily_raw, decisions)

        # Показуємо результат після виправлення
        issues_after = detect_issues(daily)
        score_after, label_after = get_quality_score(issues_after)
        st.success(f"✅ Після виправлення: {score_after}/100 — {label_after}")

    # Статистика датасету
    st.divider()
    st.subheader("📊 Статистика датасету")
    stats = get_stats(daily)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Кількість днів", stats['days'])
    c2.metric("Початок", stats['start'])
    c3.metric("Кінець", stats['end'])
    c4.metric("Середня виручка", f"{currency}{stats['mean']}")
    c5.metric("Макс. виручка", f"{currency}{stats['max']}")

# ========== ВКЛАДКА 2 — АНАЛІТИКА ==========
with tab2:
    st.subheader("Аналітика бізнесу")

    general = get_general_stats(daily, currency)

    # Загальні показники
    st.markdown("### 📈 Загальні показники")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Загальна виручка", f"{currency}{general['total']:,.0f}")
    m2.metric("Середня денна виручка", f"{currency}{general['mean']}")
    trend_delta = f"{'+' if general['trend_pct'] > 0 else ''}{general['trend_pct']}%"
    m3.metric("Тренд", general['trend_direction'].capitalize(), delta=trend_delta)
    m4.metric("Найкращий день", general['best_day']['date'],
              delta=f"{currency}{general['best_day']['value']}")

    # Дні тижня
    weekday = get_weekday_stats(daily)
    if weekday:
        st.markdown("### 📅 По днях тижня")
        wd1, wd2 = st.columns(2)
        with wd1:
            st.success(f"✅ **Найкращий день:** {weekday['best']['name']} — "
                      f"{currency}{weekday['best']['value']:.0f}")
        with wd2:
            st.error(f"❌ **Найслабший день:** {weekday['worst']['name']} — "
                    f"{currency}{weekday['worst']['value']:.0f}")

        # Графік по днях тижня
        fig_wd = go.Figure(go.Bar(
            x=list(weekday['averages'].keys()),
            y=list(weekday['averages'].values()),
            marker_color=['#FF6B35' if v == weekday['best']['value']
                         else '#2E86AB'
                         for v in weekday['averages'].values()]
        ))
        fig_wd.update_layout(
            title="Середня виручка по днях тижня",
            xaxis_title="День тижня",
            yaxis_title=f"Виручка ({currency})",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_wd, use_container_width=True)
    else:
        st.info("📅 Для аналізу по днях тижня потрібно мінімум 14 днів даних.")

    # Місяці
    monthly = get_monthly_stats(daily)
    if monthly:
        st.markdown("### 🗓️ По місяцях")
        mo1, mo2 = st.columns(2)
        with mo1:
            st.success(f"✅ **Найкращий місяць:** {monthly['best']['name']} — "
                      f"{currency}{monthly['best']['value']:.0f}")
        with mo2:
            st.error(f"❌ **Найслабший місяць:** {monthly['worst']['name']} — "
                    f"{currency}{monthly['worst']['value']:.0f}")

        fig_mo = go.Figure(go.Bar(
            x=list(monthly['averages'].keys()),
            y=list(monthly['averages'].values()),
            marker_color=['#FF6B35' if v == monthly['best']['value']
                         else '#2E86AB'
                         for v in monthly['averages'].values()]
        ))
        fig_mo.update_layout(
            title="Середня виручка по місяцях",
            xaxis_title="Місяць",
            yaxis_title=f"Виручка ({currency})",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_mo, use_container_width=True)
    else:
        st.info("🗓️ Для місячної аналітики потрібно мінімум 60 днів даних.")

    # Порівняння періодів
    comparison = get_comparison_stats(daily)
    if comparison:
        st.markdown("### ⚖️ Порівняння періодів")
        p1, p2, p3 = st.columns(3)
        p1.metric(
            f"Перша половина ({comparison['first_period']['start']} – {comparison['first_period']['end']})",
            f"{currency}{comparison['first_period']['mean']}/день"
        )
        p2.metric(
            f"Друга половина ({comparison['second_period']['start']} – {comparison['second_period']['end']})",
            f"{currency}{comparison['second_period']['mean']}/день",
            delta=f"{'+' if comparison['change_pct'] > 0 else ''}{comparison['change_pct']}%"
        )
        p3.metric(
            "Вихідні vs будні",
            f"+{comparison['weekend_boost_pct']}%" if comparison['weekend_boost_pct'] > 0
            else f"{comparison['weekend_boost_pct']}%"
        )
    else:
        st.info("⚖️ Для порівняльної аналітики потрібно мінімум 30 днів даних.")

    # Висновки
    st.markdown("### 💡 Висновки для вашого бізнесу")
    insights = generate_insights(daily, currency)
    for insight in insights:
        st.write(f"• {insight}")

# ========== ВКЛАДКА 3 — ПРОГНОЗ ==========
with tab3:
    st.subheader("Прогноз попиту")

    run_button = st.button("🚀 Запустити прогноз", type="primary")

    if run_button:
        if len(daily) < 14:
            st.error(f"❌ Для побудови прогнозу потрібно мінімум 14 днів. У вас {len(daily)} днів після обробки.")
            st.stop()

        with st.spinner("Навчання моделі та побудова прогнозу..."):
            try:
                train, test = split_train_test(daily, test_days=min(30, len(daily) // 5))
                _, forecast_test = train_and_forecast(train, periods=len(test), country=country_code)
                metrics = calculate_metrics(test, forecast_test)
                model, forecast = train_and_forecast(daily, periods=periods, country=country_code)
            except Exception as e:
                st.error(f"❌ Помилка при побудові прогнозу: {str(e)}")
                st.stop()

        st.success("Прогноз готовий!")

        # Метрики
        st.markdown("### 📈 Якість моделі")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{currency}{metrics['MAE']}", help="Середня абсолютна похибка")
        m2.metric("RMSE", f"{currency}{metrics['RMSE']}", help="Середньоквадратична похибка")
        m3.metric("MAPE", f"{metrics['MAPE']}%", help="Середня відсоткова похибка")

        summary = generate_summary(metrics, currency)
        st.info(f"💡 {summary}")

        # ---- Порівняння моделей ----
        st.markdown("### 🔬 Порівняння моделей")

        with st.spinner("Порівнюємо Prophet з ARIMA та Baseline..."):
            from src.comparison import compare_models

            comparison_results = compare_models(daily, test_days=min(30, len(daily) // 5))

        comparison_data = []
        for model_name, m in comparison_results.items():
            if m['MAE'] is not None:
                comparison_data.append({
                    'Модель': model_name,
                    'MAE': f"{currency}{m['MAE']}",
                    'RMSE': f"{currency}{m['RMSE']}",
                    'MAPE': f"{m['MAPE']}%"
                })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Висновок
        prophet_mape = comparison_results['Prophet']['MAPE']
        baseline_mape = comparison_results['Baseline (ковзне середнє)']['MAPE']
        arima_mape = comparison_results['ARIMA']['MAPE']

        if arima_mape and baseline_mape:
            best_alt = min(baseline_mape, arima_mape)
            best_alt_name = "Baseline" if baseline_mape < arima_mape else "ARIMA"
            improvement = round(best_alt - prophet_mape, 1)
            if improvement > 0:
                st.success(
                    f"✅ Prophet перевершує альтернативні методи: "
                    f"точність краща на {improvement}% від {best_alt_name}."
                )
            else:
                st.info(
                    f"ℹ️ На цьому датасеті {best_alt_name} показує кращий результат "
                    f"(MAPE {best_alt}% vs Prophet {prophet_mape}%). "
                    f"Це може бути пов'язано з малим обсягом даних або специфікою датасету."
                )

        # Графік прогнозу
        st.markdown("### 📉 Прогноз виручки")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily['ds'], y=daily['y'],
            mode='markers', name='Фактична виручка',
            marker=dict(color='#2E86AB', size=5, opacity=0.7)
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255, 165, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Довірчий інтервал'
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Прогноз',
            line=dict(color='#FF6B35', width=2)
        ))

        last_date = daily['ds'].max().strftime('%Y-%m-%d')
        fig.add_shape(
            type="line", x0=last_date, x1=last_date,
            y0=0, y1=1, yref="paper",
            line=dict(color="gray", dash="dash")
        )
        fig.add_annotation(
            x=last_date, y=1, yref="paper",
            text="Початок прогнозу", showarrow=False,
            yshift=10, font=dict(color="gray")
        )

        fig.update_layout(
            xaxis_title="Дата",
            yaxis_title=f"Виручка ({currency})",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Компоненти моделі
        st.markdown("### 🔍 Компоненти моделі")
        col_trend, col_weekly = st.columns(2)

        with col_trend:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['trend'],
                mode='lines', line=dict(color='#2E86AB', width=2)
            ))
            fig_trend.update_layout(
                title="Тренд", xaxis_title="Дата",
                yaxis_title=f"Виручка ({currency})", height=300
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_weekly:
            days_ua = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд']
            weekly_cols = [c for c in forecast.columns if 'weekly' in c]
            if weekly_cols:
                weekly_data = forecast[['ds'] + weekly_cols].copy()
                weekly_data['weekday'] = weekly_data['ds'].dt.dayofweek
                weekly_avg = weekly_data.groupby('weekday')[weekly_cols[0]].mean()

                fig_weekly = go.Figure(go.Bar(
                    x=days_ua, y=weekly_avg.values,
                    marker_color='#FF6B35'
                ))
                fig_weekly.update_layout(
                    title="Тижнева сезонність",
                    xaxis_title="День тижня",
                    yaxis_title="Відхилення від тренду",
                    height=300
                )
                st.plotly_chart(fig_weekly, use_container_width=True)

        # Таблиця прогнозу
        st.markdown("### 📅 Деталі прогнозу")
        forecast_display = forecast[forecast['ds'] > daily['ds'].max()][
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        ].copy()
        forecast_display.columns = ['Дата', 'Прогноз', 'Мінімум', 'Максимум']
        forecast_display['Дата'] = forecast_display['Дата'].dt.strftime('%d.%m.%Y')
        for col in ['Прогноз', 'Мінімум', 'Максимум']:
            forecast_display[col] = forecast_display[col].apply(
                lambda x: f"{currency}{x:.2f}"
            )
        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

        # Експорт
        st.markdown("### 💾 Експорт результатів")
        e1, e2 = st.columns(2)

        with e1:
            csv = forecast_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Завантажити прогноз (CSV)",
                data=csv, file_name="forecast.csv",
                mime="text/csv", use_container_width=True
            )

        with e2:
            fig_bytes = fig.to_image(format="png", width=1200, height=500, scale=2)
            st.download_button(
                label="⬇️ Завантажити графік (PNG)",
                data=fig_bytes, file_name="forecast_plot.png",
                mime="image/png", use_container_width=True
            )