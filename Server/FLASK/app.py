# ===== NUMPY COMPATIBILITY WORKAROUND =====
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy.core'] = np.core
    sys.modules['numpy.core._multiarray_umath'] = np.core._multiarray_umath
# ===== END WORKAROUND =====

# TensorFlow setup
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Explicitly disable GPU

# Flask + others
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

# Your custom resume analyzers
from resume_analyzer.analyzer import analyze_resume
from AnalyzeResume.resume_analyzer import analyze_resume1

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- ROUTES -------- #

@app.route('/tf-info')
def tf_info():
    return jsonify({
        "version": tf.__version__,
        "devices": [d.device_type for d in tf.config.list_physical_devices()],
        "using_gpu": bool(tf.config.list_physical_devices('GPU'))
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = analyze_resume(filepath)
    return jsonify(result)


@app.route('/analyze-resume', methods=['POST'])
def analyze_resume_route():
    if 'file' not in request.files:   # 🔹 fixed mismatch (was "resume" before)
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = analyze_resume1(filepath)
    return jsonify(result)


# -------- MODEL PREDICTION -------- #
# Lazy load model so TF doesn’t blow memory at startup
model = None
emotion_labels = ['anger', 'neutral', 'fear', 'sad', 'disgust', 'happy', 'surprise']

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model("models/best_model.h5")

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(request.files['image']).convert('L')
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]

    return jsonify({'emotion': emotion})


# -------- MAIN ENTRY -------- #
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # 🔹 dynamic port for Render
    app.run(host="0.0.0.0", port=port, debug=False)
