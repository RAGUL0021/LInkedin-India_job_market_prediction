#!/usr/bin/env bash
pip install -r requirements.txt
[ -f models/best_model.joblib ] || python src/train.py
(sleep 2; python -m webbrowser http://127.0.0.1:5000) &
python app.py
