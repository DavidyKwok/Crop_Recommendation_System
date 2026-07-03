@echo off
python train_yield_model.py
if errorlevel 1 pause && exit /b 1
python rank_crops.py
if errorlevel 1 pause && exit /b 1
streamlit run app.py
