#!/bin/bash

source venv/Scripts/activate

echo "** Module Knockout IOs Tests **"
cd knockout_ios && pytest . -x -n auto --disable-warnings
