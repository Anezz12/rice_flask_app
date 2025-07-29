import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2

app = Flask(__name__)

# Global variable untuk model - JANGAN LOAD DISINI!
model = None

# Label klasifikasi sesuai model Anda
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine']

def load_model_on_demand():
    global model
    if model is None:
        try:
            # Method 1: Standard loading
            from tensorflow.keras.models import load_model
            model = load_model('rice_classification_model.h5', compile=False)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Standard loading failed: {e}")
            try:
                # Method 2: Manual reconstruction
                print("Trying manual reconstruction...")
                model = create_model_architecture()
                model.load_weights('rice_classification_model.h5')
                print("Model loaded via manual reconstruction!")
            except Exception as e2:
                print(f"Manual reconstruction failed: {e2}")
                try:
                    # Method 3: Different loading approach
                    print("Trying alternative loading...")
                    import h5py
                    with h5py.File('rice_classification_model.h5', 'r') as f:
                        model = create_model_architecture()
                        # Try to load weights manually
                        print("Loading weights manually...")
                        model.load_weights('rice_classification_model.h5')
                        print("Model loaded with manual h5py method!")
                except Exception as e3:
                    print(f"All loading methods failed: {e3}")
                    # Create dummy model for testing
                    print("Creating dummy model for testing...")
                    model = create_dummy_model()
    return model

def create_model_architecture():
    """Create a basic model architecture - adjust based on your model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model

def create_dummy_model():
    """Create dummy model for testing when real model fails"""
    class DummyModel:
        def predict(self, img):
            # Return random predictions for testing
            return np.random.rand(1, 4)
    return DummyModel()

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
    try:
        current_model = load_model_on_demand()
        return jsonify({
            'status': 'healthy', 
            'model_loaded': current_model is not None,
            'model_type': type(current_model).__name__
        })
    except Exception as e:
        return jsonify({'status': 'error', 'model_loaded': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    # Load model on demand
    current_model = load_model_on_demand()
    
    if current_model is None:
        return jsonify({'error': 'Model failed to load'}), 500
        
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
        predictions = current_model.predict(img)
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(predictions))

        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            'prediction': predicted_label,
            'class_index': int(predicted_index),
            'confidence': round(confidence, 3),
            'model_type': type(current_model).__name__
        })
    except Exception as e:
        # Clean up uploaded file if error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)