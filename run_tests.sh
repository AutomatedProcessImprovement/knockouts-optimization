source venv/Scripts/activate
pip install -r requirements.txt

cd knockout_ios && pytest . -x -n auto
