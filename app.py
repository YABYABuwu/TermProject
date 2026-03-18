import pathlib
import platform
import os
import pickle
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# หลอก Windows ให้รู้จัก PosixPath จาก Linux (สำหรับโหลดโมเดลข้าม OS)
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# AutoGluon optional loader
try:
    from autogluon.timeseries import TimeSeriesPredictor

    AUTOGLUON_AVAILABLE = True
except ImportError:
    TimeSeriesPredictor = None
    AUTOGLUON_AVAILABLE = False


# --- 1. ฟังก์ชันโหลดโมเดล ---
def load_submission_model(model_dir="submission_models"):
    if not AUTOGLUON_AVAILABLE:
        print("AutoGluon not installed; using fallback simulator.")
        return None
    try:
        if os.path.isdir(model_dir):
            return TimeSeriesPredictor.load(model_dir)
        model_file = os.path.join(model_dir, "predictor.pkl")
        if os.path.exists(model_file):
            try:
                return TimeSeriesPredictor.load(model_dir)
            except Exception:
                with open(model_file, "rb") as f:
                    return pickle.load(f)
    except Exception as error:
        print(f"Error loading model from '{model_dir}': {error}")
    return None


# --- 2. เตรียมข้อมูลดิบและ Metric ---
def get_combined_data():
    tickers = {
        "^GSPC": "S&P 500",
        "GC=F": "Gold",
        "CL=F": "Crude Oil",
        "NG=F": "Natural Gas",
        "BTC-USD": "Bitcoin",
    }
    raw_data = yf.download(list(tickers.keys()), period="150d", interval="1d")["Close"]
    df_prices = raw_data.ffill().dropna()

    df_corr = df_prices.pct_change().corr().round(2)
    df_norm = (df_prices / df_prices.iloc[0]) * 100

    df_sp500 = df_prices[["^GSPC"]].copy().reset_index()
    df_sp500.columns = ["Date", "Close"]
    df_sp500["Yesterday_Close"] = df_sp500["Close"].shift(1)

    # Fallback Simulator (กรณีไม่มีโมเดล)
    np.random.seed(42)
    df_sp500["Pred"] = df_sp500["Yesterday_Close"].apply(
        lambda x: x * np.random.uniform(0.992, 1.010) if pd.notna(x) else np.nan
    )
    df_sp500 = df_sp500.dropna().copy()
    df_sp500["Daily_Return"] = df_sp500["Close"].pct_change() * 100
    return df_prices, df_corr, df_norm, df_sp500, tickers


def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    true_diff = np.diff(y_true) > 0
    pred_diff = np.diff(y_pred) > 0
    dir_acc = np.mean(true_diff == pred_diff) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    max_err = np.max(np.abs(y_true - y_pred))

    return {
        "MAPE": f"{mape:.2f}%",
        "MAE": f"${mae:.2f}",
        "RMSE": f"{rmse:.2f}",
        "Directional Accuracy": f"{dir_acc:.2f}%",
        "R² Score": f"{r2:.4f}",
        "Max Error": f"${max_err:.2f}",
    }


# --- 3. ฟังก์ชันพยากรณ์และ Backtest ---
def get_model_forecast(
    df_prices, predictor, target_ticker="^GSPC", prediction_length=7
):
    if predictor is None:
        return None
    try:
        df_train = df_prices.reset_index().rename(
            columns={df_prices.index.name or "Date": "timestamp"}
        )
        df_train = df_train.melt(
            id_vars="timestamp", var_name="item_id", value_name="target"
        )
        forecast_df = predictor.predict(df_train)
        series = (
            forecast_df.loc[target_ticker]
            if target_ticker in forecast_df.index.get_level_values(0)
            else forecast_df.iloc[:prediction_length]
        )
        return pd.DataFrame(
            {"Date": series.index, "Model_Forecast": series["mean"].values}
        )
    except Exception as exc:
        print(f"Forecast failed: {exc}")
        return None


def get_model_backtest_metrics(
    df_prices, predictor, prediction_length=28, target_ticker="^GSPC"
):
    global df_backtest_results  # เพื่อส่งค่าออกไปพล็อตกราฟ
    if predictor is None or len(df_prices) <= prediction_length:
        return None
    try:
        input_data = df_prices.iloc[:-prediction_length]
        actual_values = df_prices.iloc[-prediction_length:][target_ticker].values
        all_preds = []
        current_df = input_data.copy()

        steps = prediction_length // 7
        for i in range(steps):
            train_ag = current_df.reset_index().rename(
                columns={current_df.index.name or "Date": "timestamp"}
            )
            train_ag = train_ag.melt(
                id_vars="timestamp", var_name="item_id", value_name="target"
            )

            single_pred = predictor.predict(train_ag)
            week_preds = single_pred.loc[target_ticker]["mean"].values
            all_preds.extend(week_preds)

            # เลื่อน Window ข้อมูลจริงเข้าไปเพื่อทายรอบถัดไป (Rolling)
            start_idx = len(input_data) + (i * 7)
            actual_chunk = df_prices.iloc[start_idx : start_idx + 7]
            current_df = pd.concat([current_df, actual_chunk])

        final_preds = np.array(all_preds[:prediction_length])
        df_backtest_results = pd.DataFrame(
            {"Date": df_prices.index[-prediction_length:], "Pred": final_preds}
        )
        return calculate_metrics(actual_values, final_preds)
    except Exception as exc:
        print(f"Rolling backtest failed: {exc}")
        return None


# --- 4. EXECUTION (ลำดับการทำงาน) ---
df_backtest_results = pd.DataFrame()
df_prices, df_corr, df_norm, df_sp500, ticker_map = get_combined_data()
predictor = load_submission_model("submission_models")

# รัน Backtest และ Forecast
model_backtest_metrics = get_model_backtest_metrics(
    df_prices, predictor, prediction_length=28
)
model_forecast_df = get_model_forecast(df_prices, predictor)

# เลือก Metric แสดงผล
if model_backtest_metrics:
    metrics = model_backtest_metrics
    metrics["Status"] = "✅ AI Model Active"
else:
    metrics = calculate_metrics(df_sp500["Close"].values, df_sp500["Pred"].values)
    metrics["Status"] = "⚠️ Using Fallback Simulator"

# เตรียมข้อมูลกราฟ
main_forecast_data = [
    go.Scatter(
        x=df_sp500["Date"],
        y=df_sp500["Close"],
        name="Actual Price",
        line={"color": "#0984e3", "width": 2.5},
    )
]

if not df_backtest_results.empty:
    main_forecast_data.append(
        go.Scatter(
            x=df_backtest_results["Date"],
            y=df_backtest_results["Pred"],
            name="AI 4-Week Backtest",
            line={"color": "#6c5ce7", "dash": "dot"},
        )
    )
else:
    main_forecast_data.append(
        go.Scatter(
            x=df_sp500["Date"],
            y=df_sp500["Pred"],
            name="Randomized Fallback",
            line={"color": "#b2bec3", "dash": "dot"},
        )
    )

if model_forecast_df is not None:
    main_forecast_data.append(
        go.Scatter(
            x=model_forecast_df["Date"],
            y=model_forecast_df["Model_Forecast"],
            name="AI 7-Day Forecast",
            line={"color": "#fa8231", "dash": "dash"},
        )
    )

# --- 5. DASH LAYOUT ---
app = dash.Dash(__name__)
app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, sans-serif",
        "backgroundColor": "#f0f2f5",
        "padding": "25px",
    },
    children=[
        html.Div(
            [
                html.H1(
                    "Multi-Asset & AI Intelligence Dashboard",
                    style={"textAlign": "center", "color": "#1a2a6c"},
                ),
                html.P(
                    "Integrated Analysis: S&P 500 Rolling Backtest and Asset Performance",
                    style={"textAlign": "center", "color": "#636e72"},
                ),
            ]
        ),
        # KPI Cards
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "center",
                "gap": "15px",
                "marginBottom": "25px",
            },
            children=[
                html.Div(
                    [
                        html.P(
                            k,
                            style={
                                "margin": "0",
                                "fontSize": "13px",
                                "color": "#636e72",
                                "fontWeight": "600",
                            },
                        ),
                        html.H3(
                            v,
                            style={
                                "margin": "5px 0",
                                "color": "#2d3436",
                                "fontSize": "18px",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "15px",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                        "minWidth": "140px", 
                        "textAlign": "center",
                        "flex": "1",
                    },
                )
                for k, v in metrics.items()
            ],
        ),
        # Main Graph
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "15px",
                "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                "marginBottom": "25px",
            },
            children=[
                html.H3(
                    "Target Focus: S&P 500 (Actual vs AI Rolling Prediction)",
                    style={"color": "#0984e3", "marginLeft": "10px"},
                ),
                dcc.Graph(
                    figure={
                        "data": main_forecast_data,
                        "layout": go.Layout(
                            template="plotly_white", hovermode="x unified", height=450
                        ),
                    }
                ),
            ],
        ),
        # Asset Performance & Correlation
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.1fr 0.9fr",
                "gap": "20px",
                "marginBottom": "25px",
            },
            children=[
                html.Div(
                    [
                        html.H4(
                            "Asset Performance Growth (%)", style={"padding": "10px"}
                        ),
                        dcc.Graph(
                            figure=px.line(
                                df_norm.rename(columns=ticker_map),
                                labels={"value": "Growth %"},
                            ).update_layout(template="plotly_white", height=380)
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "15px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "Inter-Asset Correlation Matrix", style={"padding": "10px"}
                        ),
                        dcc.Graph(
                            figure=px.imshow(
                                df_corr.rename(index=ticker_map, columns=ticker_map),
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                            ).update_layout(height=380)
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "15px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                    },
                ),
            ],
        ),
        # Footer
        html.Div(
            style={
                "backgroundColor": "#2d3436",
                "padding": "25px",
                "borderRadius": "15px",
                "color": "white",
                "textAlign": "center",
            },
            children=[
                html.H4(
                    "Latest Support Asset Market Prices",
                    style={"color": "#dfe6e9", "marginBottom": "15px"},
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-around",
                        "flexWrap": "wrap",
                    },
                    children=[
                        html.Div(
                            [
                                html.Small(ticker_map[t]),
                                html.H3(f"${df_prices[t].iloc[-1]:,.2f}"),
                            ]
                        )
                        for t in ["GC=F", "CL=F", "NG=F", "BTC-USD"]
                    ],
                ),
            ],
        ),
        html.P(
            f"Last Intelligence Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style={
                "textAlign": "center",
                "marginTop": "20px",
                "color": "#b2bec3",
                "fontSize": "12px",
            },
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
