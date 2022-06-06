REM Create & activate virtual environment
py -m venv venv
.\venv\Scripts\activate

REM Install dependencies
python -m pip install --upgrade pip
python -m pip install pytest wheel
pip install -r requirements.txt
choco install graphviz

REM Test the installation
SET RUNNING_TESTS=true
cd knockout_ios
pytest -m "pipeline" --disable-warnings

pause