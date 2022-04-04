import os
import sys

from flask import Flask, redirect, url_for, render_template, session, request, flash, get_flashed_messages
from werkzeug.utils import secure_filename

current_directory = os.getcwd()  

sys.path.insert(0, current_directory+'../aisrc')

UPLOAD_FOLDER = '~/Downloads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

models = [{"name":"CNN", "acc":97},{"name":"CNN Class Weighted with Noise", "acc":85},{"name":"LSTM","acc":83.5}]
fields = ["Non-ecotic (Normal) Beats", "Supraventricular Ectopic Beats", "Ventricular Ectopic Beats", "Fusion Beats","Unknown Beats"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def use_model(input, modelnum):
    print(input)

    
    results = [[0.1,0.1,0.6,0.1,0.1],[0.5,0.2,0.1,0.1,0.1]]
    resultsposneg = []
    for l in results:
        x = max(l)
        r = []
        for i in l:
            if i == x:
                r.append("+")
            else:
                r.append("-")
        resultsposneg.append(r)

    return(resultsposneg)

@app.route("/")
def home():
    error = request.args.get('error')
    print(error)
    print("home")
    return render_template("input.html", error = error)

@app.route("/results")
def conic():
    print("data")
    return render_template("results.html", model_name="sample" , model_accuracy=99.5, normal = True )


@app.route("/checker", methods = ['POST'])
def check():
    print("DATA:")
    model = int(request.form['model'])
    if "file" in request.files.keys() and request.files['file'].filename != "":
        if allowed_file(request.files['file'].filename):
            f = request.files['file']
            model_info = models[model-1]
            # vvvvvvv do something with the file data vvvvvvv
            data = use_model(f.read(),model-1)
            # ^^^^^^^ do something with the file data ^^^^^^^

            return render_template("results.html", model_name=model_info["name"] , model_accuracy=model_info["acc"], fields=fields, results = data )
        return redirect(url_for('home', error= "File type not supported."))
    return redirect(url_for('home', error= "No file selected."))

if __name__ == "__main__":
    app.debug = True

app.run()
