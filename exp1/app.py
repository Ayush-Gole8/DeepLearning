import os
import io

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = os.path.join("outputs", "models", "best_model.keras")

# The exact class names from the dataset (alphabetical order)
CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

IMG_SIZE = (256, 256)

# Map each class to one of the 3 municipal bins used by the frontend.
# Keys here match the bin identifiers in static/script.js.
BIN_MAPPING = {
    'battery':    'non-recyclable',
    'biological': 'biodegradable',
    'cardboard':  'recyclable',
    'clothes':    'recyclable',
    'glass':      'recyclable',
    'metal':      'recyclable',
    'paper':      'recyclable',
    'plastic':    'recyclable',
    'shoes':      'recyclable',
    'trash':      'non-recyclable',
}

BIN_LABELS = {
    'biodegradable':  'Biodegradable',
    'recyclable':     'Recyclable',
    'non-recyclable': 'Non-recyclable / Hazardous',
}

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}

# ---------------------------------------------------------
# 2. LOAD MODEL ONCE AT STARTUP
# ---------------------------------------------------------
model = None
model_error = None

print("Loading model. This might take a few seconds...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")
except Exception as e:
    model_error = str(e)
    print(f"Error loading model: {e}")
    print("Make sure you have trained the model so 'best_model.keras' exists.")

# ---------------------------------------------------------
# 3. FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload cap


def preprocess_image(file_bytes):
    """Decode raw bytes -> (1, 256, 256, 3) float32 array.

    The trained model applies MobileNetV2 `preprocess_input` internally, so we
    feed raw 0-255 values here (matching predict.py), without extra scaling.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # (1, 256, 256, 3)
    return img_batch


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Guard: model must be available.
    if model is None:
        return jsonify({
            'error': 'Model is not available. Check the server logs.',
            'details': model_error,
        }), 503

    # Guard: a file must be present in the request.
    if 'image' not in request.files:
        return jsonify({'error': 'No image was uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            'error': f'Unsupported file type "{ext}". Please upload an image.'
        }), 400

    try:
        file_bytes = file.read()
        img_batch = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({'error': f'Could not read the image: {e}'}), 400

    try:
        predictions = model.predict(img_batch, verbose=0)
        probs = predictions[0]

        predicted_index = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(probs) * 100.0)

        bin_id = BIN_MAPPING.get(predicted_class, 'non-recyclable')
        bin_label = BIN_LABELS[bin_id]

        # Full per-class breakdown for a richer UI (sorted high -> low).
        breakdown = sorted(
            [
                {'class': CLASS_NAMES[i], 'confidence': round(float(p) * 100.0, 2)}
                for i, p in enumerate(probs)
            ],
            key=lambda d: d['confidence'],
            reverse=True,
        )

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'bin_id': bin_id,
            'bin_label': bin_label,
            'breakdown': breakdown,
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
