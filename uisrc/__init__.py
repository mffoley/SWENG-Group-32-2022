from flask import Flask, redirect, url_for, render_template, session, request, flash, get_flashed_messages

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("welcome.html", name="kevin"  )

@app.route("/conic")
def conic():
    print("data")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.debug = True

app.run()
