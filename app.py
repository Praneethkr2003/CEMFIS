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
CLASSES = ["S1", "S2", "S3", "S4"]
CLASS_DESC = {
    "S1": "Healthy — physical state healthy and relay ON",
    "S2": "Fault — physical fault with relay OFF",
    "S3": "Hidden fault — physical fault while relay remains ON",
    "S4": "Relay fault — physical state healthy while relay is OFF",
}
# Synthetic assumptions used by Dataset V2. These are NOT confirmed field specifications.
NOMINAL = {"relay_coil_voltage_V": 5.0, "motor_speed_rpm": 56.0, "motor_voltage_V": 12.0}
TOLERANCE = {"relay_coil_voltage_V": 0.20, "motor_speed_rpm": 0.20, "motor_voltage_V": 0.20}

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

def validate_sequence(g):
    g = g.sort_values("timestamp_ms").copy()
    missing = [c for c in FEATURES if c not in g.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if len(g) != 1001:
        raise ValueError(f"Expected exactly 1,001 rows, got {len(g)}")
    ts = g["timestamp_ms"].to_numpy()
    if ts[0] != 0 or ts[-1] != 10000 or not np.all(np.diff(ts) == 10):
        raise ValueError("timestamp_ms must run from 0 to 10,000 in 10 ms steps.")
    x = g[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Telemetry contains missing or non-finite values.")
    if not np.isin(g["relay_status"].to_numpy(), [0, 1]).all():
        raise ValueError("relay_status must contain only 0 or 1.")
    return g

def prepare_sequence(g):
    g = validate_sequence(g)
    x = g[FEATURES].to_numpy(dtype=np.float32)
    x[:, 1:] = scaler.transform(x[:, 1:])
    return x, g

def predict_sequence(x):
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None, ...]))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs

def maintenance_risk(g):
    """Transparent prototype maintenance-risk indicator.

    This is NOT a trained predictive-maintenance model. It estimates current
    maintenance risk from deviation and instability of synthetic telemetry.
    It must be replaced/validated with maintenance-event labels before being
    presented as a predictive ML result.
    """
    scores = {}
    for col, nominal in NOMINAL.items():
        mean_val = float(g[col].mean())
        std_val = float(g[col].std())
        rel_dev = abs(mean_val - nominal) / nominal
        rel_std = std_val / nominal
        scores[col] = float(np.clip(0.7 * rel_dev + 0.3 * rel_std, 0, 1))
    physical_score = float(np.mean(list(scores.values())))
    relay_transition_rate = float(np.mean(np.abs(np.diff(g.relay_status.to_numpy(dtype=float)))))
    relay_score = min(relay_transition_rate * 10.0, 1.0)
    risk_score = float(np.clip(0.8 * physical_score + 0.2 * relay_score, 0, 1))
    if risk_score < 0.15:
        level = "LOW"
        action = "Continue monitoring"
    elif risk_score < 0.35:
        level = "MEDIUM"
        action = "Inspect trend and schedule routine inspection"
    elif risk_score < 0.60:
        level = "HIGH"
        action = "Schedule maintenance inspection"
    else:
        level = "CRITICAL"
        action = "Inspect equipment before continued operation"
    return {
        "score": risk_score,
        "level": level,
        "action": action,
        "component_scores": scores,
        "relay_instability": relay_score,
    }

def plot_telemetry(g):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.timestamp_ms, y=g.relay_coil_voltage_V, name="Coil voltage (V)"))
    fig.add_trace(go.Scatter(x=g.timestamp_ms, y=g.motor_speed_rpm, name="Motor speed (RPM)", yaxis="y2"))
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Time (ms)",
                      yaxis=dict(title="Voltage (V)"), yaxis2=dict(title="RPM", overlaying="y", side="right"),
                      legend=dict(orientation="h"))
    return fig

def build_manual_sequence(relay, coil, rpm, motor_v):
    return pd.DataFrame({
        "timestamp_ms": np.arange(0, 10001, 10, dtype=int),
        "relay_status": relay,
        "relay_coil_voltage_V": coil,
        "motor_speed_rpm": rpm,
        "motor_voltage_V": motor_v,
    })

def render_prediction(scenario, probs):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Scenario", scenario)
    c2.metric("Confidence", f"{probs.max():.1%}")
    c3.metric("S3 probability", f"{probs[2]:.1%}")
    c4.metric("S4 probability", f"{probs[3]:.1%}")
    if scenario == "S1":
        st.success("S1 — HEALTHY")
    elif scenario == "S3":
        st.error("S3 — HIDDEN FAULT: physical fault while relay remains ON")
    elif scenario == "S4":
        st.warning("S4 — RELAY FAULT: physical state healthy while relay is OFF")
    else:
        st.error("S2 — FAULT: physical fault with relay OFF")
    st.write(CLASS_DESC[scenario])

st.title("🚦 RDPMA — Point Machine Condition & Maintenance Monitor")
st.caption("GRU inference prototype • Dataset V2 • 10 ms sampling • 10,000 ms sequence")

with st.sidebar:
    st.header("Telemetry schema")
    st.code("timestamp_ms\nrelay_status\nrelay_coil_voltage_V\nmotor_speed_rpm\nmotor_voltage_V")
    st.warning("Prototype trained on synthetic Dataset V2. Not a railway safety-certified system.")
    st.info("Maintenance is currently a transparent risk-indicator prototype, not a trained predictive-maintenance model.")

input_tab, manual_tab, about_tab = st.tabs(["📁 CSV / Machine Data", "⌨️ Manual Input", "ℹ️ About"])

prepared = {}
results = []

with input_tab:
    st.subheader("Upload Point Machine telemetry")
    st.write("Upload one 10-second sequence or multiple sequences using an optional `sequence_id` column.")
    uploaded = st.file_uploader("Telemetry CSV", type=["csv"], key="csv_input")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in FEATURES + ["timestamp_ms"] if c not in df.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                groups = [(str(sid), g.copy()) for sid, g in df.groupby("sequence_id", sort=False)] if "sequence_id" in df.columns else [("uploaded_sequence", df.copy())]
                for sid, g in groups:
                    try:
                        x, clean_g = prepare_sequence(g)
                        scenario, probs = predict_sequence(x)
                        results.append({"sequence_id": sid, "prediction": scenario, "confidence": float(probs.max()),
                                        "S1": float(probs[0]), "S2": float(probs[1]), "S3": float(probs[2]), "S4": float(probs[3]), "status": "OK"})
                        prepared[sid] = (clean_g, scenario, probs)
                    except Exception as e:
                        results.append({"sequence_id": sid, "prediction": "—", "confidence": 0.0,
                                        "S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0, "status": str(e)})

with manual_tab:
    st.subheader("Manual telemetry input")
    st.caption("A single reading cannot be meaningfully classified by the GRU. This option creates a constant 10-second demo sequence from your entered operating point. Use CSV/live hardware for real temporal inference.")
    c1,c2,c3,c4 = st.columns(4)
    relay = c1.selectbox("Relay status", [1, 0], format_func=lambda x: "ON (1)" if x == 1 else "OFF (0)")
    coil = c2.number_input("Coil voltage (V)", min_value=0.0, max_value=20.0, value=5.0, step=0.01)
    rpm = c3.number_input("Motor speed (RPM)", min_value=0.0, max_value=200.0, value=56.0, step=0.1)
    motor_v = c4.number_input("Motor voltage (V)", min_value=0.0, max_value=30.0, value=12.0, step=0.01)
    if st.button("Run GRU on manual demo sequence", type="primary"):
        manual_g = build_manual_sequence(relay, coil, rpm, motor_v)
        try:
            x, clean_g = prepare_sequence(manual_g)
            scenario, probs = predict_sequence(x)
            prepared["manual_demo"] = (clean_g, scenario, probs)
            st.session_state["manual_result"] = (clean_g, scenario, probs)
        except Exception as e:
            st.error(str(e))
    if "manual_result" in st.session_state:
        g, scenario, probs = st.session_state["manual_result"]
        render_prediction(scenario, probs)
        st.plotly_chart(plot_telemetry(g), use_container_width=True)

# Display CSV results if available
if results:
    res = pd.DataFrame(results)
    valid = res[res.status == "OK"]
    st.subheader("Condition prediction")
    if len(valid) == 1:
        row = valid.iloc[0]
        render_prediction(row.prediction, np.array([row.S1,row.S2,row.S3,row.S4], dtype=float))
    else:
        st.dataframe(valid[["sequence_id","prediction","confidence","S1","S2","S3","S4"]], use_container_width=True)
    if len(valid) == 0:
        st.error("No valid 1,001-sample sequences were found.")
    else:
        selected = st.selectbox("Sequence for detailed analysis", list(prepared.keys()), key="sequence_select")
        g, scenario, probs = prepared[selected]
        left,right = st.columns([2,1])
        with left:
            st.plotly_chart(plot_telemetry(g), use_container_width=True)
        with right:
            st.metric("Relay state (majority)", "ON" if g.relay_status.mean() >= 0.5 else "OFF")
            st.metric("Coil voltage", f"{g.relay_coil_voltage_V.mean():.2f} V")
            st.metric("Motor speed", f"{g.motor_speed_rpm.mean():.2f} RPM")
            st.metric("Motor voltage", f"{g.motor_voltage_V.mean():.2f} V")
        st.subheader("Model probabilities")
        prob_df = pd.DataFrame({"Scenario": CLASSES, "Probability": probs})
        st.bar_chart(prob_df.set_index("Scenario"))

        st.subheader("Predictive maintenance")
        mr = maintenance_risk(g)
        m1,m2,m3 = st.columns(3)
        m1.metric("Maintenance risk", mr["level"])
        m2.metric("Risk score", f"{mr['score']:.1%}")
        m3.metric("Recommended action", mr["action"])
        st.caption("Prototype heuristic based on deviation/instability from Dataset V2 synthetic nominal values. This is NOT a trained maintenance prediction.")
        comp = pd.DataFrame({"Signal": list(mr["component_scores"].keys()), "Risk contribution": list(mr["component_scores"].values())})
        st.bar_chart(comp.set_index("Signal"))
        st.write("Relay instability contribution:", f"{mr['relay_instability']:.1%}")

        csv = res.to_csv(index=False).encode("utf-8")
        st.download_button("Download condition predictions", data=csv, file_name="rdpma_predictions.csv", mime="text/csv")

with about_tab:
    st.subheader("What this deployment currently does")
    st.markdown("""
- **Condition classification:** GRU predicts S1/S2/S3/S4 from a 1,001-sample Point Machine sequence.
- **CSV input:** supports one or multiple `sequence_id` sequences.
- **Manual input:** creates a constant sequence for demonstration only.
- **Maintenance:** currently provides a transparent heuristic risk indicator. It is intentionally **not** presented as a trained predictive-maintenance model.
- **Future live mode:** the same 10-second buffer can later be populated directly from STM32/hardware telemetry instead of CSV.
""")
    st.subheader("Maintenance ML next step")
    st.write("To make maintenance genuinely predictive, we need maintenance-event/degradation labels (for example inspection/repair events or time-to-failure) and must train/evaluate a separate maintenance model without target leakage.")
