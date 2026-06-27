from flask import Flask, render_template, request
from model import predict_heart_disease

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        chest_pain = int(request.form["chest_pain"])
        bp = int(request.form["bp"])
        chol = int(request.form["chol"])
        sugar = int(request.form["sugar"])

        result = predict_heart_disease(age, gender, chest_pain, bp, chol, sugar)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
