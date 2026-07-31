import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "Clean_IRO_GPS_2010-13.csv"
TEST_FILE = "Clean_IRO_GPS_2024-25.csv"

OUTPUT_DIR = "simple_rnn_digital_twin_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = ["lat", "lon"]
TARGETS = ["lat", "lon"]

SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

ROLL_OUT_STEPS = 200   # number of future points to simulate per seal

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# DISTANCE AND METRICS
# ============================================================

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in kilometres.
    """

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

    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_metrics(y_true, y_pred):
    """
    Computes trajectory prediction metrics.
    y_true and y_pred must be arrays of shape [N, 2],
    where columns are latitude and longitude.
    """

    errors_km = haversine_distance_km(
        y_true[:, 0],
        y_true[:, 1],
        y_pred[:, 0],
        y_pred[:, 1]
    )

    ade = np.mean(errors_km)
    fde = errors_km[-1]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "ADE_km": ade,
        "FDE_km": fde,
        "RMSE_deg": rmse,
        "MAE_deg": mae,
        "Median_Error_km": np.median(errors_km),
        "P90_Error_km": np.percentile(errors_km, 90),
        "P95_Error_km": np.percentile(errors_km, 95),
        "Max_Error_km": np.max(errors_km),
        "Pct_Error_gt_2km": 100.0 * np.mean(errors_km > 2.0),
        "Pct_Error_gt_5km": 100.0 * np.mean(errors_km > 5.0),
        "Pct_Error_gt_10km": 100.0 * np.mean(errors_km > 10.0),
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_gps_file(path):
    df = pd.read_csv(path)

    required_cols = ["seal", "d_date", "lat", "lon"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[required_cols].copy()

    df["d_date"] = pd.to_datetime(df["d_date"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["seal", "d_date", "lat", "lon"])
    df = df.sort_values(["seal", "d_date"]).reset_index(drop=True)

    return df


print("Loading data...")
train_df = load_gps_file(TRAIN_FILE)
test_df = load_gps_file(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Training seals: {train_df['seal'].nunique()}")
print(f"Testing seals: {test_df['seal'].nunique()}")


# ============================================================
# SCALING
# ============================================================

scaler = MinMaxScaler()

train_values = train_df[FEATURES].values
scaler.fit(train_values)

train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

train_df_scaled[FEATURES] = scaler.transform(train_df[FEATURES].values)
test_df_scaled[FEATURES] = scaler.transform(test_df[FEATURES].values)


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences_by_seal(df, seq_length=30):
    X_all = []
    y_all = []
    seal_all = []
    date_all = []

    for seal_id, group in df.groupby("seal"):
        group = group.sort_values("d_date").reset_index(drop=True)

        values = group[FEATURES].values
        dates = group["d_date"].values

        if len(group) <= seq_length:
            continue

        for i in range(len(group) - seq_length):
            X_all.append(values[i:i + seq_length])
            y_all.append(values[i + seq_length])
            seal_all.append(seal_id)
            date_all.append(dates[i + seq_length])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    return X_all, y_all, np.array(seal_all), np.array(date_all)


X_train, y_train, train_seals, train_dates = create_sequences_by_seal(
    train_df_scaled,
    SEQ_LENGTH
)

X_test, y_test, test_seals, test_dates = create_sequences_by_seal(
    test_df_scaled,
    SEQ_LENGTH
)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)


# ============================================================
# MODEL
# ============================================================

class UnivariateRNN(nn.Module):
    def __init__(
        self,
        input_size=2,
        hidden_size=64,
        num_layers=1,
        output_size=2,
        dropout=0.0
    ):
        super(UnivariateRNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh"
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred


model = UnivariateRNN(
    input_size=len(FEATURES),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=len(TARGETS),
    dropout=DROPOUT
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)


# ============================================================
# TRAINING
# ============================================================

train_losses = []

print("\nTraining RNN digital twin engine...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_loss = np.mean(epoch_losses)
    train_losses.append(mean_loss)

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {mean_loss:.8f}")


# Save training curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss: Univariate RNN Digital Twin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=300)
plt.close()


# ============================================================
# ONE-STEP PREDICTION
# ============================================================

model.eval()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

one_step_metrics = compute_metrics(y_test_inv, y_pred_inv)

print("\nOne-step prediction metrics:")
for k, v in one_step_metrics.items():
    print(f"{k}: {v:.4f}")

pd.DataFrame([one_step_metrics]).to_csv(
    os.path.join(OUTPUT_DIR, "one_step_metrics.csv"),
    index=False
)


# ============================================================
# DIGITAL TWIN AUTOREGRESSIVE ROLLOUT
# ============================================================

def autoregressive_rollout(model, initial_window_scaled, rollout_steps):
    """
    Simulates future trajectory using the model's own predictions.

    initial_window_scaled: array with shape [SEQ_LENGTH, 2]
    returns: array with shape [rollout_steps, 2]
    """

    model.eval()

    current_window = initial_window_scaled.copy()
    simulated_points = []

    with torch.no_grad():
        for _ in range(rollout_steps):
            x = torch.tensor(
                current_window[None, :, :],
                dtype=torch.float32
            ).to(device)

            pred = model(x).cpu().numpy()[0]
            simulated_points.append(pred)

            current_window = np.vstack([
                current_window[1:],
                pred.reshape(1, -1)
            ])

    simulated_points = np.array(simulated_points)
    return simulated_points


def run_digital_twin_for_each_seal(test_df_scaled, test_df_original, rollout_steps=200):
    all_rollout_metrics = []

    for seal_id, group_scaled in test_df_scaled.groupby("seal"):
        group_scaled = group_scaled.sort_values("d_date").reset_index(drop=True)

        group_original = test_df_original[
            test_df_original["seal"] == seal_id
        ].sort_values("d_date").reset_index(drop=True)

        if len(group_scaled) <= SEQ_LENGTH + 5:
            continue

        actual_rollout_steps = min(
            rollout_steps,
            len(group_scaled) - SEQ_LENGTH
        )

        initial_window_scaled = group_scaled[FEATURES].values[:SEQ_LENGTH]

        simulated_scaled = autoregressive_rollout(
            model,
            initial_window_scaled,
            actual_rollout_steps
        )

        simulated_inv = scaler.inverse_transform(simulated_scaled)

        actual_future = group_original[FEATURES].values[
            SEQ_LENGTH:SEQ_LENGTH + actual_rollout_steps
        ]

        actual_dates = group_original["d_date"].values[
            SEQ_LENGTH:SEQ_LENGTH + actual_rollout_steps
        ]

        metrics = compute_metrics(actual_future, simulated_inv)
        metrics["seal"] = seal_id
        metrics["rollout_steps"] = actual_rollout_steps

        all_rollout_metrics.append(metrics)

        # Save per-seal rollout CSV
        rollout_df = pd.DataFrame({
            "seal": seal_id,
            "date": actual_dates,
            "actual_lat": actual_future[:, 0],
            "actual_lon": actual_future[:, 1],
            "simulated_lat": simulated_inv[:, 0],
            "simulated_lon": simulated_inv[:, 1],
        })

        rollout_df.to_csv(
            os.path.join(OUTPUT_DIR, f"digital_twin_rollout_seal_{seal_id}.csv"),
            index=False
        )

        # Plot per-seal trajectory
        plt.figure(figsize=(8, 7))

        plt.plot(
            actual_future[:, 1],
            actual_future[:, 0],
            marker="o",
            markersize=2,
            linewidth=1,
            label="Actual trajectory"
        )

        plt.plot(
            simulated_inv[:, 1],
            simulated_inv[:, 0],
            marker="x",
            markersize=2,
            linewidth=1,
            label="Digital twin simulated trajectory"
        )

        plt.scatter(
            initial_window_scaled[:, 1],
            initial_window_scaled[:, 0],
            s=8,
            label="Initial observed window"
        )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Simple RNN Digital Twin Rollout: Seal {seal_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"digital_twin_rollout_seal_{seal_id}.png"),
            dpi=300
        )

        plt.close()

    metrics_df = pd.DataFrame(all_rollout_metrics)

    if len(metrics_df) > 0:
        metrics_df = metrics_df[
            [
                "seal",
                "rollout_steps",
                "ADE_km",
                "FDE_km",
                "RMSE_deg",
                "MAE_deg",
                "Median_Error_km",
                "P90_Error_km",
                "P95_Error_km",
                "Max_Error_km",
                "Pct_Error_gt_2km",
                "Pct_Error_gt_5km",
                "Pct_Error_gt_10km",
            ]
        ]

    return metrics_df


rollout_metrics_df = run_digital_twin_for_each_seal(
    test_df_scaled,
    test_df,
    rollout_steps=ROLL_OUT_STEPS
)

rollout_metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "digital_twin_rollout_metrics_per_seal.csv"),
    index=False
)

print("\nDigital twin rollout metrics per seal:")
print(rollout_metrics_df.head())

print("\nMean digital twin rollout metrics:")
print(
    rollout_metrics_df[
        [
            "ADE_km",
            "FDE_km",
            "RMSE_deg",
            "MAE_deg",
            "Median_Error_km",
            "P90_Error_km",
            "P95_Error_km",
            "Max_Error_km",
            "Pct_Error_gt_2km",
            "Pct_Error_gt_5km",
            "Pct_Error_gt_10km",
        ]
    ].mean()
)


# ============================================================
# GLOBAL ONE-STEP PLOT
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    y_test_inv[:, 1],
    y_test_inv[:, 0],
    s=4,
    alpha=0.5,
    label="Actual"
)

plt.scatter(
    y_pred_inv[:, 1],
    y_pred_inv[:, 0],
    s=4,
    alpha=0.5,
    label="One-step RNN prediction"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("One-step Prediction: Actual vs RNN Predicted Locations")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "one_step_actual_vs_predicted.png"),
    dpi=300
)

plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "features": FEATURES,
        "seq_length": SEQ_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    },
    os.path.join(OUTPUT_DIR, "simple_rnn_digital_twin_model.pt")
)

print(f"\nAll outputs saved in: {OUTPUT_DIR}")
