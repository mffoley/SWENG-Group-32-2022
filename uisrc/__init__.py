import os
import sys
from flask import Flask, redirect, url_for, render_template, session, request, flash, get_flashed_messages
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '~/Downloads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def use_model(input, modelnum):
    return {"normal":True,'model_name':"Sample Model",'model_acc':99.5}




@app.route("/")
def home():
    print("home")
    return render_template("input.html", name="kevin"  )

@app.route("/results")
def conic():
    print("data")
    return render_template("results.html", model_name="sample" , model_accuracy=99.5, normal = True )


@app.route("/checker", methods = ['POST'])
def check():
    print(list(request.form.keys()))
    if "file" in request.files.keys() and allowed_file(request.files['file'].filename):
        f = request.files['file']

        # vvvvvvv do something with the file data vvvvvvv
        data = use_model(f.read(),1)
        # ^^^^^^^ do something with the file data ^^^^^^^

        return render_template("results.html", model_name=data["model_name"] , model_accuracy=data["model_acc"], normal = data["normal"] )

    return redirect(url_for('home'))

if __name__ == "__main__":
    app.debug = True

app.run()
