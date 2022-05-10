#!/bin/bash

source venv/Scripts/activate

pip install -r requirements.txt

echo "** Module Knockout IOs Tests **"
cd knockout_ios && pytest . -x -n auto --disable-warnings
