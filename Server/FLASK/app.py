import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Flask imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Setup upload directory
UPLOAD_FOLDER = Path('/tmp/uploads')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# ===== ROUTES ===== #

@app.route('/')
def home():
    return "Resume Analyzer and Emotion Detection API is running! Use /analyze, /analyze-resume, or /predict endpoints."

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        from resume_analyzer.analyzer import analyze_resume
    except ImportError as e:
        return jsonify({'error': f'Dependency error: {str(e)}'}), 500  
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = analyze_resume(filepath)
    return jsonify(result)

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume_route():
    try:
        from AnalyzeResume.resume_analyzer import analyze_resume1
    except ImportError as e:
        return jsonify({'error': f'Dependency error: {str(e)}'}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = analyze_resume1(filepath)
    return jsonify(result)


# -------- MODEL PREDICTION -------- #
emotion_labels = ['anger', 'neutral', 'fear', 'sad', 'disgust', 'happy', 'surprise']

# Use a function closure instead of global variables
def get_model():
    model = None
    def load():
        nonlocal model
        if model is None:
            # Configure TensorFlow memory growth FIRST
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            physical_devices = tf.config.list_physical_devices('CPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Now load the model
            from tensorflow.keras.models import load_model
            model = load_model("models/best_model.h5")
        return model
    return load

model_loader = get_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        model = model_loader()
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

    image = Image.open(request.files['image']).convert('L')
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]

    return jsonify({'emotion': emotion})


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 7860))  # use HF's port when provided
    # create uploads dir
    os.makedirs('uploads', exist_ok=True)
    from waitress import serve  # optional local runner; Docker will use gunicorn
    serve(app, host='0.0.0.0', port=port)
