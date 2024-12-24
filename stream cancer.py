from flask import Flask, request, jsonify # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API Prediksi Kanker berjalan!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Ambil data input
        features = np.array(data["features"]).reshape(1, -1)
        # Prediksi
        prediction = model.predict(features)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)