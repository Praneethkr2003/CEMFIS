# RDPMA Point Machine Deployment

A Streamlit deployment of the current Point Machine GRU inference model.

## What it does

- Accepts Point Machine telemetry as CSV.
- Supports one sequence or multiple `sequence_id` groups.
- Requires 1,001 samples from 0–10,000 ms at 10 ms intervals.
- Runs the trained GRU model.
- Returns S1/S2/S3/S4 prediction and class probabilities.
- Visualizes telemetry and basic sequence statistics.
- Exports predictions as CSV.

## Required CSV columns

```text
timestamp_ms
relay_status
relay_coil_voltage_V
motor_speed_rpm
motor_voltage_V
```

Optional: `sequence_id` to upload multiple sequences in one file.

## Important

This is a research/POC deployment trained on synthetic Dataset V2. It is **not** a railway safety-certified system and the current GRU does **not** produce a separately trained maintenance prediction.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push this folder to GitHub, then deploy `app.py` using Streamlit Community Cloud.
