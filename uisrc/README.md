#### Install and run on localhost
1. Navigate to desired directory and clone the repo
2. Make a new venv (instructions below)
3. Install requirements by runnning `$ pip3 install -r requirements.txt`
4. Run the app by running `$ python3 __init__.py`
  1. Flask app will be running on `http://127.0.0.1:5000/`
  2. Control-C to end app


#### Steps to create a venv:

1. In a terminal, go to the folder in which you want to keep your venv
2. Run python3 -m venv EXVENV
  1. We are using EXVENV as the name of the virtual enviornment; you can use any name you would like
3. Activate your virtual enviornment by running source EXVENV/bin/activate
  1. Your computer's name will now be preceded by (EXVENV). You are now inside of the virtual enviornment.
4. Install dependencies 
5. To exit the venv, run deactivate