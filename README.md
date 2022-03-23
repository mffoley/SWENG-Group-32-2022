
#### Steps to create a venv:

1. In a terminal, go to the folder in which you want to keep your venv
2. Run python3 -m venv EXVENV
  1. We are using EXVENV as the name of the virtual enviornment; you can use any name you would like
3. Activate your virtual enviornment by running source EXVENV/bin/activate
  1. Your computer's name will now be preceded by (EXVENV). You are now inside of the virtual enviornment.
4. Install dependencies by typing `$ pip3 install -r requirements.txt`
5. To exit the venv, run deactivate

- To create a new requirements document (does not need to be done except when a new dependency is added) use  `$ pip3 freeze > requirements.txt`


#### BY WED 30 MAR, HAVE WEBPAGE DONE
Upload file, dropdown for model, "classify" runs model
Sampling frequency specified, expect the person to upload it
all local
