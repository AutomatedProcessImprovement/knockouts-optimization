#!/bin/bash

source venv/Scripts/activate

echo "** Module Knockout IOs Tests **"
export RUNNING_TESTS=true
cd knockout_ios && pytest . -x -n auto --disable-warnings
