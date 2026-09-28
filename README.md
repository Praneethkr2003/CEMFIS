# RDPMA Point Machine Deployment

A Streamlit deployment of the current Point Machine GRU inference model with CSV input, manual demo input, telemetry visualization, and a transparent maintenance-risk prototype.

## Current capabilities

- Upload Point Machine telemetry as CSV.
- Supports one sequence or multiple `sequence_id` groups.
- Requires 1,001 samples from 0–10,000 ms at 10 ms intervals.
- Runs the trained Point Machine GRU.
- Returns S1/S2/S3/S4 prediction and probabilities.
- Visualizes coil voltage and motor speed over time.
- Manual operating-point input can create a constant 10-second **demo** sequence.
- Provides a transparent maintenance-risk indicator based on deviation/instability from synthetic V2 nominal values.
- Exports condition predictions as CSV.

## Required CSV columns

```text
timestamp_ms
relay_status
relay_coil_voltage_V
motor_speed_rpm
motor_voltage_V
```

Optional: `sequence_id` to upload multiple sequences in one file.

## Important maintenance note

The current application does **not** claim to have a trained predictive-maintenance model. The maintenance section is a transparent prototype risk indicator. A genuine predictive-maintenance model requires maintenance-event/degradation labels (inspection, repair, failure, time-to-failure, etc.) and independent evaluation.

## Important model note

This is a research/POC deployment trained on synthetic Dataset V2. It is **not a railway safety-certified system** and must not be used for operational railway control.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push this folder to GitHub, then deploy `app.py` using Streamlit Community Cloud.
