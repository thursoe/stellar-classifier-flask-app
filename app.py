from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import classifier
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

env_config = os.getenv("PROD_APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

model = joblib.load('model_stellar.pkl')

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data['Abs Mag'] = data.pop('AbsMag', None)
    data['Spectral Class'] = data.pop('SpectralClass', None)
    prediction = classifier.make_predict(model, data)
    return jsonify({'predictions': [prediction]})


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)

        required_columns = ['Temp', 'L', 'R', 'Abs Mag', 'Spectral Class']
        if not all(column in df.columns for column in required_columns):
            return jsonify({"error": "Missing required columns in the CSV file"}), 400

        data_dict = df.to_dict(orient='records')
        predictions = [classifier.make_predict(
            model, record) for record in data_dict]

        return jsonify({'predictions': predictions}), 200

    return jsonify({"error": "Invalid file type"}), 400


if __name__ == '__main__':
    app.run()
