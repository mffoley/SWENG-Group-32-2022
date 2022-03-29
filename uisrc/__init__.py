import os
import sys
from flask import Flask, redirect, url_for, render_template, session, request, flash, get_flashed_messages
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '~/Downloads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

models = [{"name":"Simple Model", "acc":99.5},{"name":"Complicated Model", "acc":92}]
fields = ["normal", "var2", "var3", "var4","var5"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def use_model(input, modelnum):
    print(input)
    return [{"normal":True,"var2":False,"var3":True,"var4":False,"var5":False},{"normal":True,"var2":False,"var3":True,"var4":False,"var5":False}]


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
