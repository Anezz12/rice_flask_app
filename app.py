import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load model
try:
    model = load_model('rice_classification_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Label klasifikasi sesuai model Anda
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine']

def preprocess_image(img_path):
    # Model expects (64, 64, 3) input shape
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Resize to 64x64
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension: (1, 64, 64, 3)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create static directory if it doesn't exist
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    file_path = os.path.join('static', file.filename)
    
    try:
        file.save(file_path)
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(predictions))

        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            'prediction': predicted_label,
            'class_index': int(predicted_index),
            'confidence': round(confidence, 3)
        })
    except Exception as e:
        # Clean up uploaded file if error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)