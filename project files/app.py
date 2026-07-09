from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ==========================
# Load the Trained ML Model
# ==========================
model = pickle.load(open("model/hdi_model.pkl", "rb"))

# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("home.html")


# ==========================
# Prediction Page
# ==========================
@app.route("/predict")
def predict():
    return render_template("indexnew.html")


# ==========================
# Prediction Result
# ==========================
@app.route("/result", methods=["GET", "POST"])
def result():
    try:
        # Get values from form
        life = float(request.form["life"])
        expected = float(request.form["expected"])
        mean = float(request.form["mean"])
        gni = float(request.form["gni"])

        # Prepare input for model
        data = np.array([[life, expected, mean, gni]])

        # Predict
        prediction = float(model.predict(data)[0])

        # Decide HDI Category
        if prediction >= 0.800:
            category = "Very High HDI"
        elif prediction >= 0.700:
            category = "High HDI"
        elif prediction >= 0.550:
            category = "Medium HDI"
        else:
            category = "Low HDI"

        return render_template(
            "result.html",
            prediction=round(prediction, 3),
            category=category
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error",
            category=str(e)
        )


# ==========================
# Run Flask App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)