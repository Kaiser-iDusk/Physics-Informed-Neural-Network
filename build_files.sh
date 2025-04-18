# create venv
echo "Venv create mode start"
python3.9 -m venv venv
echo "Venv create mode end"

# activate the virtual environment
echo "Venv Activation..."
source venv/bin/activate
echo "Venv Activated!"

# upgrade pip
echo "Upgrading pip..."
/vercel/path0/venv/bin/python3.9 -m pip install --upgrade pip
echo "Pip upgraded!"

# install requirements
echo "Installing requirements..."
pip install -r requirements.txt
echo "Requirements Installed!"

# build static
echo "Building static..."
python3.9 manage.py collectstatic --noinput
echo "Static built!"