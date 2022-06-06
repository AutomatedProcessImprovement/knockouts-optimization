echo "Creating & activating virtual environment"
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies"
python -m pip install --upgrade pip
python -m pip install pytest wheel
pip install -r requirements.txt
sudo apt install graphviz

echo "Testing the installation"
export RUNNING_TESTS=true
cd knockout_ios
pytest -m "pipeline" --disable-warnings
