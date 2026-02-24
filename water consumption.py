import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Water Consumption Forecast", layout="wide")
st.title("💧 Malaysia Water Consumption Forecasting Dashboard")

# ================= DATA LOADING =================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\user\Downloads\water_consumption.csv")

    df['date'] = pd.to_datetime(df['date'])   # ✅ FIXED
    df['year'] = df['date'].dt.year

    return df

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("Filter Settings")

selected_state = st.sidebar.selectbox("Select State", df['state'].unique())
selected_sector = st.sidebar.selectbox("Select Sector", df['sector'].unique())

test_size = 5
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 10, 5)

# ================= FILTER DATA =================
full_series = df[
    (df['state'] == selected_state) &
    (df['sector'] == selected_sector)
].sort_values('year')

if len(full_series) < test_size + 5:
    st.warning("Not enough data for this selection.")
    st.stop()

train_df = full_series.iloc[:-test_size]
test_df = full_series.iloc[-test_size:]

train_series = pd.Series(train_df['value'].values, index=train_df['year'])

# ================= METRICS =================
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ================= ML FORECAST FUNCTION =================
def get_ml_forecast(model_type, train_data, test_len, forecast_len):

    temp_df = train_data.copy()

    temp_df['lag_1'] = temp_df['value'].shift(1)
    temp_df['lag_2'] = temp_df['value'].shift(2)
    temp_df['rolling_mean_3'] = temp_df['value'].rolling(3).mean()
    temp_df = temp_df.dropna()

    features = ['year', 'lag_1', 'lag_2', 'rolling_mean_3']

    X = temp_df[features]
    y = temp_df['value']

    if model_type == 'XGB':
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    model.fit(X, y)

    results = []
    last_row = temp_df.iloc[-1].copy()

    for _ in range(test_len + forecast_len):

        new_year = last_row['year'] + 1

        new_row = pd.DataFrame([[

            new_year,
            last_row['value'],
            last_row['lag_1'],
            np.mean([last_row['value'], last_row['lag_1'], last_row['lag_2']])

        ]], columns=features)

        pred = model.predict(new_row)[0]
        results.append(pred)

        last_row['year'] = new_year
        last_row['lag_2'] = last_row['lag_1']
        last_row['lag_1'] = last_row['value']
        last_row['value'] = pred

    return results[:test_len], results[test_len:]


# ================= MODELS =================

with st.spinner("Optimizing ARIMA..."):
    arima_model = pm.auto_arima(train_series, seasonal=False, stepwise=True,d=0, suppress_warnings=True)
    arima_test = arima_model.predict(n_periods=test_size)
    arima_fore = arima_model.predict(n_periods=test_size + forecast_years)[test_size:]

des_model = ExponentialSmoothing(train_series, trend='add').fit()
des_test = des_model.forecast(test_size)
des_fore = des_model.forecast(test_size + forecast_years)[test_size:]

xgb_test, xgb_fore = get_ml_forecast('XGB', train_df, test_size, forecast_years)
rf_test, rf_fore = get_ml_forecast('RF', train_df, test_size, forecast_years)

# ================= PLOT =================
col1, col2 = st.columns([2, 1])

future_years = np.arange(test_df['year'].max() + 1,
                         test_df['year'].max() + 1 + forecast_years)

with col1:

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(train_df['year'], train_df['value'], label='Train', color='black')
    ax.plot(test_df['year'], test_df['value'], linestyle=':', color='black', label='Test')

    plot_data = [
        (arima_test, arima_fore, 'blue', 'ARIMA'),
        (des_test, des_fore, 'red', 'DES'),
        (xgb_test, xgb_fore, 'green', 'XGBoost'),
        (rf_test, rf_fore, 'orange', 'Random Forest')
    ]

    for test_pred, fore_pred, color, label in plot_data:
        ax.plot(test_df['year'], test_pred, '--', color=color, alpha=0.6)
        ax.plot(future_years, fore_pred, color=color, label=label)

    ax.legend()
    st.pyplot(fig)

with col2:
    pred_df = pd.DataFrame({
        "Year": future_years,
        "ARIMA": arima_fore,
        "DES": des_fore,
        "XGBoost": xgb_fore,
        "Random Forest": rf_fore
    }).set_index("Year")

    st.dataframe(pred_df.style.format("{:.2f}"))

# ================= PERFORMANCE =================
st.subheader("Model Performance Evaluation")

def get_metrics(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted)),
        calculate_mape(actual, predicted)
    )

actual_vals = test_df['value'].values

metrics_dict = {
    "ARIMA": get_metrics(actual_vals, arima_test),
    "DES": get_metrics(actual_vals, des_test),
    "XGBoost": get_metrics(actual_vals, xgb_test),
    "Random Forest": get_metrics(actual_vals, rf_test)
}

perf_df = pd.DataFrame(metrics_dict, index=["MAE", "RMSE", "MAPE (%)"]).T

perf_df['Rank_MAE'] = perf_df['MAE'].rank()
perf_df['Rank_RMSE'] = perf_df['RMSE'].rank()
perf_df['Rank_MAPE'] = perf_df['MAPE (%)'].rank()

perf_df['Average_Rank'] = perf_df[['Rank_MAE', 'Rank_RMSE', 'Rank_MAPE']].mean(axis=1)
perf_df['Ranking'] = perf_df['Average_Rank'].rank().astype(int)

perf_df = perf_df.sort_values("Ranking")

st.table(perf_df[["MAE", "RMSE", "MAPE (%)", "Ranking"]].style.format({
    "MAE": "{:.2f}",
    "RMSE": "{:.2f}",
    "MAPE (%)": "{:.2f}%"
}))