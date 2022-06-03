#!/bin/bash

source venv/Scripts/activate
pip install -r requirements.txt

python -Wignore -m streamlit run main-ui.py
