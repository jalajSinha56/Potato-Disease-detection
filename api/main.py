from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app, origins=["http://localhost", "http://localhost:3000"], supports_credentials=True)

# Load the model
MODEL = tf.keras.models.load_model("./saved_models/1")

# Class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Health check route
@app.route("/ping", methods=["GET"])
def ping():
    return "Hello, I am alive"

# Helper function to read image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return jsonify({
        'class': predicted_class,
        'confidence': confidence
    })

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)

