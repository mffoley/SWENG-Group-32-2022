#### Install and run on localhost
1. Navigate to desired directory and clone the repo
2. Make a new venv (instructions on main readme)
3. Install requirements by runnning `$ pip3 install -r ../requirements.txt`
4. Run the app by running `$ python3 __init__.py`
  1. Flask app will be running on `http://127.0.0.1:5000/`
  2. Control-C to end app


### Git steps
1. `git clone git@github.com:tiscovsa/SWENG-Group-32-2022.git` (only done once)
2. Make edits
3. `git pull` (do this frequently)
  1. If you pull and there is a merge conflict, it will show a vi window with text. This is not important unless there is a conflict you have to address (this will be very obvious though), type `:q` and press enter to exit. 
4. `git status` (shows you all changes you have made recently)
5. `git add <file names>` (note: only add files you have actively changed, do NOT add venv files)
6. `git commit -m '<description of changes you made>'`
7. `git push`