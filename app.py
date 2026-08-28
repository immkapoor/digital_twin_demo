import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn

import joblib
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

# For local testing, you can use:
TEST_FILE = "Clean_IRO_GPS_2010-13.csv"

# For public deployment, use an anonymized sample file:
# TEST_FILE = "sample_Clean_IRO_GPS_2024-25.csv"

SCALER_FILE = "scaler_window_10.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# AVAILABLE MODELS
# ============================================================

AVAILABLE_MODELS = {
    "RNN window 10": {
        "path": "RNN_window_10.pt",
        "type": "RNN",
        "seq_length": 10
    },
    "LSTM window 10": {
        "path": "LSTM_window_10.pt",
        "type": "LSTM",
        "seq_length": 10
    },
    "Transformer window 10": {
        "path": "Transformer_window_10.pt",
        "type": "Transformer",
        "seq_length": 10
    },
}


# ============================================================
# MODEL DEFINITIONS
# These match your original training file.
# ============================================================

class RNNModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, output_dim=2):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_dim=2,
        dim_feedforward=2048
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])


# ============================================================
# DISTANCE AND METRICS
# ============================================================

def haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def compute_rollout_metrics(actual, predicted):
    errors = haversine_distance_km(
        actual[:, 0],
        actual[:, 1],
        predicted[:, 0],
        predicted[:, 1]
    )

    return {
        "ADE_km": float(np.mean(errors)),
        "FDE_km": float(errors[-1]),
        "Median_Error_km": float(np.median(errors)),
        "P90_Error_km": float(np.percentile(errors, 90)),
        "P95_Error_km": float(np.percentile(errors, 95)),
        "Max_Error_km": float(np.max(errors)),
        "Pct_Error_gt_2km": float(100.0 * np.mean(errors > 2.0)),
        "Pct_Error_gt_5km": float(100.0 * np.mean(errors > 5.0)),
        "Pct_Error_gt_10km": float(100.0 * np.mean(errors > 10.0)),
    }


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    df["d_date"] = pd.to_datetime(df["d_date"], errors="coerce")

    if "v_mask" in df.columns:
        df = df[df["v_mask"] == 0]

    required_cols = ["seal", "d_date", "lat", "lon"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[required_cols].copy()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=required_cols)
    df = df.sort_values(["seal", "d_date"]).reset_index(drop=True)

    df["year"] = df["d_date"].dt.year
    df["month"] = df["d_date"].dt.month
    df["month_name"] = df["d_date"].dt.strftime("%B")
    df["year_month"] = df["d_date"].dt.strftime("%Y-%m")

    return df


def build_fallback_scaler(df, features):
    scaler = StandardScaler()
    scaler.fit(df[features].values.astype(np.float32))
    return scaler


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def clean_state_dict_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_raw_state_dict(model_path):
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a raw state_dict or checkpoint dictionary."
        )

    return clean_state_dict_keys(state_dict)


def infer_rnn_hidden_layers(state_dict, model_type):
    if model_type == "RNN":
        base_key = "rnn.weight_ih_l0"
        prefix = "rnn.weight_ih_l"
        gate_multiplier = 1
    elif model_type == "LSTM":
        base_key = "lstm.weight_ih_l0"
        prefix = "lstm.weight_ih_l"
        gate_multiplier = 4
    else:
        return 64, 1

    if base_key not in state_dict:
        return 64, 1

    weight = state_dict[base_key]
    hidden_dim = int(weight.shape[0] // gate_multiplier)

    layer_indices = []
    for key in state_dict.keys():
        if key.startswith(prefix) and "_reverse" not in key:
            suffix = key.replace(prefix, "")
            try:
                layer_indices.append(int(suffix))
            except ValueError:
                pass

    num_layers = max(layer_indices) + 1 if layer_indices else 1
    return hidden_dim, num_layers


def infer_transformer_params(state_dict):
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 2048

    if "input_projection.weight" in state_dict:
        d_model = int(state_dict["input_projection.weight"].shape[0])

    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            if len(parts) > 2:
                try:
                    layer_indices.append(int(parts[2]))
                except ValueError:
                    pass

    if layer_indices:
        num_layers = max(layer_indices) + 1

    ff_keys = [
        key for key in state_dict.keys()
        if key.endswith("linear1.weight")
    ]

    if ff_keys:
        dim_feedforward = int(state_dict[ff_keys[0]].shape[0])

    for candidate in [8, 4, 2, 1]:
        if d_model % candidate == 0:
            nhead = candidate
            break

    return d_model, nhead, num_layers, dim_feedforward


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_selected_model(
    model_name,
    model_path,
    model_type,
    seq_length,
    data_for_fallback_scaler
):
    state_dict = load_raw_state_dict(model_path)

    features = ["lat", "lon"]

    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        scaler_source = f"Loaded from {SCALER_FILE}"
    else:
        scaler = build_fallback_scaler(data_for_fallback_scaler, features)
        scaler_source = "Fallback StandardScaler fitted on loaded CSV"

    if model_type == "RNN":
        hidden_dim, num_layers = infer_rnn_hidden_layers(
            state_dict,
            model_type="RNN"
        )

        model = RNNModel(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2
        ).to(DEVICE)

        model_params = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "d_model": None,
            "nhead": None,
            "dim_feedforward": None,
        }

    elif model_type == "LSTM":
        hidden_dim, num_layers = infer_rnn_hidden_layers(
            state_dict,
            model_type="LSTM"
        )

        model = LSTMModel(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=2
        ).to(DEVICE)

        model_params = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "d_model": None,
            "nhead": None,
            "dim_feedforward": None,
        }

    elif model_type == "Transformer":
        d_model, nhead, num_layers, dim_feedforward = infer_transformer_params(
            state_dict
        )

        model = TransformerModel(
            input_dim=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=2,
            dim_feedforward=dim_feedforward
        ).to(DEVICE)

        model_params = {
            "hidden_dim": None,
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        st.error(
            f"Could not load checkpoint for {model_name}.\n\n"
            f"Checkpoint path: {model_path}\n\n"
            f"First checkpoint keys:\n{list(state_dict.keys())[:25]}\n\n"
            f"Original PyTorch error:\n{error}"
        )
        st.stop()

    model.eval()

    model_info = {
        "model_name": model_name,
        "model_type": model_type,
        "features": features,
        "seq_length": seq_length,
        "model_path": model_path,
        "scaler_source": scaler_source,
        **model_params,
    }

    return model, scaler, model_info


# ============================================================
# PREDICTION MODES
# ============================================================

def predict_one_step(model, scaler, month_df, features, seq_length, start_index):
    window_start = start_index - seq_length
    window_end = start_index

    initial_window_original = month_df[features].values[window_start:window_end]
    initial_window_scaled = scaler.transform(initial_window_original).astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(
            initial_window_scaled[None, :, :],
            dtype=torch.float32
        ).to(DEVICE)

        pred_scaled = model(x).cpu().numpy()

    predicted_future = scaler.inverse_transform(pred_scaled)

    actual_future = month_df[features].values[start_index:start_index + 1]
    actual_dates = month_df["d_date"].values[start_index:start_index + 1]

    return initial_window_original, predicted_future, actual_future, actual_dates


def predict_sliding_window(
    model,
    scaler,
    month_df,
    features,
    seq_length,
    start_index,
    prediction_horizon
):
    predictions_scaled = []

    for step in range(prediction_horizon):
        current_index = start_index + step

        window_start = current_index - seq_length
        window_end = current_index

        current_window_original = month_df[features].values[window_start:window_end]
        current_window_scaled = scaler.transform(current_window_original).astype(np.float32)

        with torch.no_grad():
            x = torch.tensor(
                current_window_scaled[None, :, :],
                dtype=torch.float32
            ).to(DEVICE)

            pred_scaled = model(x).cpu().numpy()[0]
            predictions_scaled.append(pred_scaled)

    predictions_scaled = np.array(predictions_scaled)
    predicted_future = scaler.inverse_transform(predictions_scaled)

    actual_future = month_df[features].values[
        start_index:start_index + prediction_horizon
    ]

    actual_dates = month_df["d_date"].values[
        start_index:start_index + prediction_horizon
    ]

    initial_window_original = month_df[features].values[
        start_index - seq_length:start_index
    ]

    return initial_window_original, predicted_future, actual_future, actual_dates


def predict_autoregressive_rollout(
    model,
    scaler,
    month_df,
    features,
    seq_length,
    start_index,
    prediction_horizon
):
    window_start = start_index - seq_length
    window_end = start_index

    initial_window_original = month_df[features].values[window_start:window_end]
    current_window_scaled = scaler.transform(initial_window_original).astype(np.float32)

    predictions_scaled = []

    with torch.no_grad():
        for _ in range(prediction_horizon):
            x = torch.tensor(
                current_window_scaled[None, :, :],
                dtype=torch.float32
            ).to(DEVICE)

            pred_scaled = model(x).cpu().numpy()[0]
            predictions_scaled.append(pred_scaled)

            current_window_scaled = np.vstack([
                current_window_scaled[1:],
                pred_scaled.reshape(1, -1)
            ]).astype(np.float32)

    predictions_scaled = np.array(predictions_scaled)
    predicted_future = scaler.inverse_transform(predictions_scaled)

    actual_future = month_df[features].values[
        start_index:start_index + prediction_horizon
    ]

    actual_dates = month_df["d_date"].values[
        start_index:start_index + prediction_horizon
    ]

    return initial_window_original, predicted_future, actual_future, actual_dates


def run_prediction_mode(
    prediction_mode,
    model,
    scaler,
    month_df,
    features,
    seq_length,
    start_index,
    prediction_horizon
):
    if prediction_mode == "One-step prediction":
        return predict_one_step(
            model=model,
            scaler=scaler,
            month_df=month_df,
            features=features,
            seq_length=seq_length,
            start_index=start_index
        )

    if prediction_mode == "Sliding-window prediction":
        return predict_sliding_window(
            model=model,
            scaler=scaler,
            month_df=month_df,
            features=features,
            seq_length=seq_length,
            start_index=start_index,
            prediction_horizon=prediction_horizon
        )

    if prediction_mode == "Autoregressive rollout":
        return predict_autoregressive_rollout(
            model=model,
            scaler=scaler,
            month_df=month_df,
            features=features,
            seq_length=seq_length,
            start_index=start_index,
            prediction_horizon=prediction_horizon
        )

    raise ValueError(f"Unknown prediction mode: {prediction_mode}")


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def make_trajectory_plot(
    month_df,
    initial_window,
    predicted_future,
    actual_future,
    start_point,
    prediction_mode,
    selected_model_name
):
    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=month_df["lat"],
            lon=month_df["lon"],
            mode="lines+markers",
            name="Full monthly trajectory",
            marker=dict(size=5),
            line=dict(width=1),
            opacity=0.45
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=initial_window[:, 0],
            lon=initial_window[:, 1],
            mode="lines+markers",
            name="Observed input window",
            marker=dict(size=7),
            line=dict(width=3)
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=predicted_future[:, 0],
            lon=predicted_future[:, 1],
            mode="lines+markers",
            name=f"Prediction: {selected_model_name}",
            marker=dict(size=8),
            line=dict(width=4)
        )
    )

    if actual_future is not None and len(actual_future) > 0:
        fig.add_trace(
            go.Scattermapbox(
                lat=actual_future[:, 0],
                lon=actual_future[:, 1],
                mode="lines+markers",
                name="Actual future trajectory",
                marker=dict(size=7),
                line=dict(width=3)
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[start_point[0]],
            lon=[start_point[1]],
            mode="markers",
            name="Prediction start point",
            marker=dict(size=14)
        )
    )

    center_lat = month_df["lat"].mean()
    center_lon = month_df["lon"].mean()

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=7
        ),
        height=700,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        title=f"{selected_model_name} | {prediction_mode}"
    )

    return fig


def make_error_plot(result_df, prediction_mode, selected_model_name):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result_df["prediction_step"],
            y=result_df["error_km"],
            mode="lines+markers",
            name="Prediction error"
        )
    )

    fig.update_layout(
        title=f"Pointwise Error Over Horizon: {selected_model_name} | {prediction_mode}",
        xaxis_title="Prediction step",
        yaxis_title="Error distance (km)",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    return fig


def make_metric_bar_plot(metrics, prediction_mode, selected_model_name):
    metrics_plot_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics_plot_df["Metric"],
            y=metrics_plot_df["Value"],
            name="Metric value"
        )
    )

    fig.update_layout(
        title=f"Aggregate Error Metrics: {selected_model_name} | {prediction_mode}",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=450,
        margin=dict(l=40, r=20, t=60, b=130),
        xaxis=dict(tickangle=-45)
    )

    return fig


def get_metric_description_table():
    return pd.DataFrame({
        "Metric": [
            "ADE_km",
            "FDE_km",
            "Median_Error_km",
            "P90_Error_km",
            "P95_Error_km",
            "Max_Error_km",
            "Pct_Error_gt_2km",
            "Pct_Error_gt_5km",
            "Pct_Error_gt_10km"
        ],
        "Full name": [
            "Average Displacement Error",
            "Final Displacement Error",
            "Median spatial error",
            "90th percentile spatial error",
            "95th percentile spatial error",
            "Maximum spatial error",
            "Percentage of errors greater than 2 km",
            "Percentage of errors greater than 5 km",
            "Percentage of errors greater than 10 km"
        ],
        "Meaning": [
            "Average distance between the actual trajectory and the predicted trajectory across all predicted points.",
            "Distance between the final actual point and the final predicted point.",
            "Typical prediction error; less affected by extreme errors than ADE.",
            "Error value below which 90% of predicted points fall.",
            "Error value below which 95% of predicted points fall.",
            "Largest prediction error observed during the selected prediction horizon.",
            "Percentage of predicted points more than 2 km away from the real trajectory.",
            "Percentage of predicted points more than 5 km away from the real trajectory.",
            "Percentage of predicted points more than 10 km away from the real trajectory."
        ],
        "Interpretation": [
            "Lower values indicate better average trajectory fidelity.",
            "Lower values indicate less endpoint drift.",
            "Useful when a few large failures distort the average.",
            "Shows high-error behaviour while ignoring the worst 10%.",
            "Shows near-worst-case behaviour while ignoring the worst 5%.",
            "Indicates the worst prediction failure in the selected run.",
            "Shows the proportion of moderate spatial errors.",
            "Shows the proportion of large spatial errors.",
            "Shows the proportion of severe trajectory failures."
        ]
    })


def get_prediction_mode_description_table():
    return pd.DataFrame({
        "Prediction mode": [
            "One-step prediction",
            "Sliding-window prediction",
            "Autoregressive rollout"
        ],
        "Input used": [
            "Observed previous sequence only.",
            "Observed previous sequence at every prediction step.",
            "Observed initial sequence, then model-generated predictions."
        ],
        "What it evaluates": [
            "Immediate next-location prediction.",
            "Stable repeated next-step prediction over a known trajectory.",
            "Long-horizon digital twin simulation and accumulated drift."
        ],
        "Expected behaviour": [
            "Usually lowest error.",
            "Usually more stable than rollout.",
            "May drift over time because errors accumulate."
        ]
    })


def safe_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("|", "_")
        .replace(":", "_")
    )


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="Seal Movement Digital Twin",
    page_icon="🌊",
    layout="wide"
)

st.title("Interactive Seal Movement Digital Twin")

st.markdown(
    """
This interface supports multiple prediction modes and multiple trained univariate trajectory models.
The current models use latitude and longitude as input features.
"""
)


# ============================================================
# FILE CHECKS
# ============================================================

if not os.path.exists(TEST_FILE):
    st.error(f"Test data file not found: {TEST_FILE}")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = load_data(TEST_FILE)


# ============================================================
# SIDEBAR: MODEL AND MODE SELECTION
# ============================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select model",
    list(AVAILABLE_MODELS.keys())
)

selected_model_config = AVAILABLE_MODELS[selected_model_name]
selected_model_path = selected_model_config["path"]
selected_model_type = selected_model_config["type"]
selected_model_seq_length = selected_model_config["seq_length"]

if not os.path.exists(selected_model_path):
    st.sidebar.error(f"Model file not found: {selected_model_path}")
    st.stop()

model, scaler, model_info = load_selected_model(
    model_name=selected_model_name,
    model_path=selected_model_path,
    model_type=selected_model_type,
    seq_length=selected_model_seq_length,
    data_for_fallback_scaler=df
)

features = model_info["features"]
seq_length = model_info["seq_length"]

st.sidebar.header("Prediction Mode")

prediction_mode = st.sidebar.selectbox(
    "Select prediction mode",
    [
        "One-step prediction",
        "Sliding-window prediction",
        "Autoregressive rollout"
    ],
    index=2
)

st.sidebar.header("Trajectory Selection")

seal_options = sorted(df["seal"].unique())

selected_seal = st.sidebar.selectbox(
    "Select seal",
    seal_options
)

seal_df = df[df["seal"] == selected_seal].copy()
month_options = sorted(seal_df["year_month"].unique())

selected_month = st.sidebar.selectbox(
    "Select month",
    month_options
)

month_df = seal_df[seal_df["year_month"] == selected_month].copy()
month_df = month_df.sort_values("d_date").reset_index(drop=True)

st.sidebar.write(f"Points available: {len(month_df)}")
st.sidebar.write(f"Input sequence length: {seq_length}")

if len(month_df) <= seq_length + 1:
    st.warning(
        "This selected seal-month does not have enough GPS points for the trained sequence length."
    )
    st.stop()

max_start_index = len(month_df) - 1

start_index = st.sidebar.slider(
    "Select prediction start index",
    min_value=seq_length,
    max_value=max_start_index,
    value=seq_length,
    step=1
)

max_possible_horizon = len(month_df) - start_index

if max_possible_horizon <= 0:
    st.warning("No future points are available after this start index.")
    st.stop()

if prediction_mode == "One-step prediction":
    prediction_horizon = 1
    st.sidebar.info("One-step mode predicts only the next point.")
else:
    prediction_horizon = st.sidebar.slider(
        "Prediction horizon",
        min_value=1,
        max_value=max(1, min(200, max_possible_horizon)),
        value=min(30, max_possible_horizon),
        step=1
    )


# ============================================================
# RUN SELECTED PREDICTION MODE
# ============================================================

start_point = month_df[features].values[start_index]

initial_window_original, predicted_future, actual_future, actual_dates = run_prediction_mode(
    prediction_mode=prediction_mode,
    model=model,
    scaler=scaler,
    month_df=month_df,
    features=features,
    seq_length=seq_length,
    start_index=start_index,
    prediction_horizon=prediction_horizon
)


# ============================================================
# RESULT DATAFRAME
# ============================================================

result_df = pd.DataFrame({
    "prediction_step": np.arange(1, len(predicted_future) + 1),
    "date": actual_dates,
    "actual_lat": actual_future[:, 0],
    "actual_lon": actual_future[:, 1],
    "predicted_lat": predicted_future[:, 0],
    "predicted_lon": predicted_future[:, 1],
})

result_df["error_km"] = haversine_distance_km(
    result_df["actual_lat"].values,
    result_df["actual_lon"].values,
    result_df["predicted_lat"].values,
    result_df["predicted_lon"].values
)

metrics = compute_rollout_metrics(actual_future, predicted_future)


# ============================================================
# MAIN DISPLAY
# ============================================================

left_col, right_col = st.columns([2, 1])

with left_col:
    trajectory_fig = make_trajectory_plot(
        month_df=month_df,
        initial_window=initial_window_original,
        predicted_future=predicted_future,
        actual_future=actual_future,
        start_point=start_point,
        prediction_mode=prediction_mode,
        selected_model_name=selected_model_name
    )

    st.plotly_chart(trajectory_fig, use_container_width=True)

with right_col:
    st.subheader("Selected configuration")

    config_fields = [
        "Selected model",
        "Model type",
        "Prediction mode",
        "Seal",
        "Month",
        "Start index",
        "Prediction horizon",
        "Input sequence length",
        "Input features",
        "Scaler source",
        "Device"
    ]

    config_values = [
        selected_model_name,
        selected_model_type,
        prediction_mode,
        selected_seal,
        selected_month,
        start_index,
        prediction_horizon,
        seq_length,
        ", ".join(features),
        model_info["scaler_source"],
        str(DEVICE)
    ]

    if selected_model_type in ["RNN", "LSTM"]:
        config_fields.extend(["Hidden dimension", "Number of layers"])
        config_values.extend([
            model_info["hidden_dim"],
            model_info["num_layers"]
        ])

    if selected_model_type == "Transformer":
        config_fields.extend([
            "d_model",
            "nhead",
            "Number of transformer layers",
            "Feedforward dimension"
        ])
        config_values.extend([
            model_info["d_model"],
            model_info["nhead"],
            model_info["num_layers"],
            model_info["dim_feedforward"]
        ])

    config_df = pd.DataFrame({
        "Field": config_fields,
        "Value": config_values
    })

    st.dataframe(config_df, use_container_width=True)

    if model_info["scaler_source"].startswith("Fallback"):
        st.warning(
            "The original training scaler file was not found. "
            "A fallback StandardScaler was fitted on the loaded CSV. "
            "For final experiments and public deployment, keep scaler_window_10.pkl with the app."
        )

    st.subheader("Prediction start point")

    start_point_df = pd.DataFrame({
        "date": [month_df.loc[start_index, "d_date"]],
        "lat": [start_point[0]],
        "lon": [start_point[1]]
    })

    st.dataframe(start_point_df, use_container_width=True)

    st.subheader("Error summary")

    metrics_df = pd.DataFrame([metrics]).T.reset_index()
    metrics_df.columns = ["Metric", "Value"]

    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        label="Download error metrics CSV",
        data=metrics_df.to_csv(index=False),
        file_name=(
            f"metrics_{safe_filename(selected_model_name)}_"
            f"{safe_filename(prediction_mode)}_"
            f"seal_{safe_filename(selected_seal)}_"
            f"{selected_month}.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# ERROR PLOT
# ============================================================

st.subheader("Pointwise prediction error")

error_fig = make_error_plot(
    result_df=result_df,
    prediction_mode=prediction_mode,
    selected_model_name=selected_model_name
)
st.plotly_chart(error_fig, use_container_width=True)


# ============================================================
# AGGREGATE METRIC BAR PLOT
# ============================================================

st.subheader("Aggregate error metrics")

metric_bar_fig = make_metric_bar_plot(
    metrics=metrics,
    prediction_mode=prediction_mode,
    selected_model_name=selected_model_name
)
st.plotly_chart(metric_bar_fig, use_container_width=True)


# ============================================================
# MODE DESCRIPTION TABLE
# ============================================================

st.subheader("Meaning of prediction modes")

mode_description_df = get_prediction_mode_description_table()
st.dataframe(mode_description_df, use_container_width=True)

st.download_button(
    label="Download prediction mode explanation table",
    data=mode_description_df.to_csv(index=False),
    file_name="prediction_mode_explanation_table.csv",
    mime="text/csv"
)


# ============================================================
# METRIC INTERPRETATION TABLE
# ============================================================

st.subheader("Meaning of error metrics")

metric_description_df = get_metric_description_table()
st.dataframe(metric_description_df, use_container_width=True)

st.download_button(
    label="Download metric interpretation table",
    data=metric_description_df.to_csv(index=False),
    file_name="metric_interpretation_table.csv",
    mime="text/csv"
)


# ============================================================
# PREDICTION TABLE
# ============================================================

st.subheader("Prediction table")

st.dataframe(result_df, use_container_width=True)

st.download_button(
    label="Download predicted trajectory CSV",
    data=result_df.to_csv(index=False),
    file_name=(
        f"prediction_{safe_filename(selected_model_name)}_"
        f"{safe_filename(prediction_mode)}_"
        f"seal_{safe_filename(selected_seal)}_"
        f"{selected_month}.csv"
    ),
    mime="text/csv"
)


# ============================================================
# SUMMARY DOWNLOAD
# ============================================================

summary_text = f"""
Seal Movement Digital Twin Summary

Selected model: {selected_model_name}
Model type: {selected_model_type}
Prediction mode: {prediction_mode}
Seal: {selected_seal}
Month: {selected_month}
Start index: {start_index}
Prediction horizon: {prediction_horizon}
Input sequence length: {seq_length}
Input features: {features}
Scaler source: {model_info["scaler_source"]}
Model path: {selected_model_path}
Device: {DEVICE}

Model parameters:
Hidden dimension: {model_info["hidden_dim"]}
Number of layers: {model_info["num_layers"]}
d_model: {model_info["d_model"]}
nhead: {model_info["nhead"]}
Feedforward dimension: {model_info["dim_feedforward"]}

Error metrics:
{metrics_df.to_string(index=False)}
"""

st.download_button(
    label="Download simulation summary TXT",
    data=summary_text,
    file_name=(
        f"summary_{safe_filename(selected_model_name)}_"
        f"{safe_filename(prediction_mode)}_"
        f"seal_{safe_filename(selected_seal)}_"
        f"{selected_month}.txt"
    ),
    mime="text/plain"
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    """
### Interpretation

The interface separates the selected model from the selected prediction mode.

In **one-step prediction**, the model predicts only the next GPS point from the observed input sequence.

In **sliding-window prediction**, the model repeatedly predicts the next point, but each prediction uses a real observed previous window. This is useful for stable quantitative evaluation because the error does not accumulate through model-generated inputs.

In **autoregressive rollout**, the model uses its own previous predictions as future inputs. This is the closest setting to a digital twin simulation, because the model generates a trajectory forward without repeatedly relying on observed future points.

Therefore, the same trained model can be inspected in three different ways: immediate prediction, stable evaluation, and long-horizon simulation.
"""
)
