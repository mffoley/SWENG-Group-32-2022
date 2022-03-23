from flask import Flask, redirect, url_for, render_template, session, request, flash, get_flashed_messages

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("welcome.html", name="kevin"  )

@app.route("/results")
def conic():
    print("data")
    return render_template("results.html", model_name="sample" , model_accuracy=99.5, normal = True )

if __name__ == "__main__":
    app.debug = True

app.run()
