import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="RDPMA Point Machine Monitor", page_icon="🚦", layout="wide")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "point_machine_gru_best.pt"
SCALER_PATH = ROOT / "model" / "scaler.joblib"
FEATURES = ["relay_status", "relay_coil_voltage_V", "motor_speed_rpm", "motor_voltage_V"]
ANALOG = FEATURES[1:]
CLASSES = ["S1", "S2", "S3", "S4"]
CLASS_DESC = {
    "S1": "Healthy — physical state healthy and relay ON",
    "S2": "Fault — physical fault with relay OFF",
    "S3": "Hidden fault — physical fault while relay remains ON",
    "S4": "Relay fault — physical state healthy while relay is OFF",
}

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=64, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 4)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(self.drop(h[-1]))

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = GRUModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

def prepare_sequence(g):
    g = g.sort_values("timestamp_ms").copy()
    missing = [c for c in FEATURES if c not in g.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if len(g) != 1001:
        raise ValueError(f"Expected exactly 1,001 rows (0–10,000 ms at 10 ms), got {len(g)}")
    ts = g["timestamp_ms"].to_numpy()
    if ts[0] != 0 or ts[-1] != 10000 or not np.all(np.diff(ts) == 10):
        raise ValueError("timestamp_ms must run from 0 to 10,000 in 10 ms steps.")
    x = g[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Telemetry contains missing or non-finite values.")
    x[:, 1:] = scaler.transform(x[:, 1:])
    return x, g

def predict_sequence(x):
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None, ...]))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs

def plot_telemetry(g):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.timestamp_ms, y=g.relay_coil_voltage_V, name="Coil voltage (V)"))
    fig.add_trace(go.Scatter(x=g.timestamp_ms, y=g.motor_speed_rpm, name="Motor speed (RPM)", yaxis="y2"))
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Time (ms)",
                      yaxis=dict(title="Voltage (V)"), yaxis2=dict(title="RPM", overlaying="y", side="right"),
                      legend=dict(orientation="h"))
    return fig

st.title("🚦 RDPMA — Point Machine Condition Monitor")
st.caption("GRU inference prototype • Dataset V2 • 10 ms sampling • 10,000 ms sequence")

with st.sidebar:
    st.header("Input requirements")
    st.write("Upload telemetry containing exactly one or more 10-second Point Machine sequences.")
    st.code("timestamp_ms\nrelay_status\nrelay_coil_voltage_V\nmotor_speed_rpm\nmotor_voltage_V")
    st.warning("Prototype trained on synthetic Dataset V2. Not a railway safety-certified system.")

uploaded = st.file_uploader("Upload Point Machine telemetry CSV", type=["csv"])

if not uploaded:
    st.info("Upload a CSV with 1,001 samples for a sequence. If sequence_id is present, each sequence_id must contain exactly 1,001 rows.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

missing = [c for c in FEATURES + ["timestamp_ms"] if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

if "sequence_id" in df.columns:
    groups = [(str(sid), g.copy()) for sid, g in df.groupby("sequence_id", sort=False)]
else:
    groups = [("uploaded_sequence", df.copy())]

results = []
prepared = {}
for sid, g in groups:
    try:
        x, clean_g = prepare_sequence(g)
        scenario, probs = predict_sequence(x)
        results.append({
            "sequence_id": sid,
            "prediction": scenario,
            "confidence": float(probs.max()),
            "S1": float(probs[0]), "S2": float(probs[1]), "S3": float(probs[2]), "S4": float(probs[3]),
            "status": "OK",
        })
        prepared[sid] = (clean_g, scenario, probs)
    except Exception as e:
        results.append({"sequence_id": sid, "prediction": "—", "confidence": 0.0,
                        "S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0,
                        "status": str(e)})

res = pd.DataFrame(results)
valid = res[res.status == "OK"]

st.subheader("Prediction")
if len(valid) == 0:
    st.error("No valid 1,001-sample sequences were found. Check the timestamps and required columns.")
    st.dataframe(res, use_container_width=True)
    st.stop()

if len(valid) == 1:
    row = valid.iloc[0]
    scenario = row.prediction
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Scenario", scenario)
    c2.metric("Confidence", f"{row.confidence:.1%}")
    c3.metric("S3 probability", f"{row.S3:.1%}")
    c4.metric("S4 probability", f"{row.S4:.1%}")
    if scenario == "S1":
        st.success("S1 — Healthy")
    elif scenario == "S3":
        st.error("S3 — HIDDEN FAULT: physical fault while relay remains ON")
    elif scenario == "S4":
        st.warning("S4 — RELAY FAULT: physical state healthy while relay is OFF")
    else:
        st.error("S2 — FAULT: physical fault with relay OFF")
    st.write(CLASS_DESC[scenario])
else:
    st.metric("Valid sequences", len(valid))
    st.dataframe(valid[["sequence_id","prediction","confidence","S1","S2","S3","S4"]], use_container_width=True)

st.subheader("Telemetry")
selected = st.selectbox("Sequence", list(prepared.keys()))
g, scenario, probs = prepared[selected]
left,right = st.columns([2,1])
with left:
    st.plotly_chart(plot_telemetry(g), use_container_width=True)
with right:
    relay_on = float(g.relay_status.mean()) >= 0.5
    st.metric("Relay state (majority)", "ON" if relay_on else "OFF")
    st.metric("Coil voltage", f"{g.relay_coil_voltage_V.mean():.2f} V")
    st.metric("Motor speed", f"{g.motor_speed_rpm.mean():.2f} RPM")
    st.metric("Motor voltage", f"{g.motor_voltage_V.mean():.2f} V")

st.subheader("Model probabilities")
prob_df = pd.DataFrame({"Scenario": CLASSES, "Probability": probs})
st.bar_chart(prob_df.set_index("Scenario"))

st.subheader("Maintenance")
st.info("Maintenance prediction is not yet a separately trained output of this GRU. The current deployed model performs S1–S4 scenario classification only. A separate maintenance-risk head/model should be added after maintenance labels and evaluation are finalized.")

csv = res.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions", data=csv, file_name="rdpma_predictions.csv", mime="text/csv")
