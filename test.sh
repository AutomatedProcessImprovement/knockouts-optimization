#!/bin/bash

source venv/Scripts/activate

echo "** Knockout IOs Tests **"
cd knockout_ios && pytest . -x -n auto --disable-warnings

echo "** Variability Analysis Tests **"
cd ../variability_analysis && pytest . -x -n auto --disable-warnings
