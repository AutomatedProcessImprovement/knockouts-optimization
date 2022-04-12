#!/bin/bash

source venv/Scripts/activate

cd knockout_ios && pytest . -x -n auto --disable-warnings
cd ../variability_analysis && pytest . -x -n auto --disable-warnings
