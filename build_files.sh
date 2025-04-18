# create venv
python3.9 -m venv venv

# activate the virtual environment
source venv/bin/activate

# upgrade pip
/vercel/path0/venv/bin/python3.9 -m pip install --upgrade pip

# install requirements
pip install -r requirements.txt

# build static
python3.9 manage.py collectstatic --noinput