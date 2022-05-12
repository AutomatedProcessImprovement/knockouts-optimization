#!/bin/bash

source venv/Scripts/activate

pip install -r requirements.txt

echo "** Module Knockout IOs Tests **"
export RUNNING_TESTS=true
cd knockout_ios && pytest . -x -n auto --disable-warnings
