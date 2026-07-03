#!/usr/bin/env bash
set -e
python3 train_yield_model.py
python3 rank_crops.py
streamlit run app.py
