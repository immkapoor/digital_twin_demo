# Seal Movement Digital Twin Demo

This repository contains an interactive Streamlit demo for a trajectory-only digital twin of seal movement.

The current model is a univariate RNN trained on latitude-longitude sequences. Given an observed input window, the app predicts future locations using one of three modes:

- One-step prediction
- Sliding-window prediction
- Autoregressive rollout

## Inputs

- Seal ID
- Month
- Prediction start point
- Prediction horizon
- Prediction mode

## Outputs

- Interactive trajectory map
- Pointwise error curve
- ADE, FDE, percentile errors, and threshold errors
- Downloadable predicted trajectory table

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
