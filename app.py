from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/predict": {"origins": "*"}})

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom unpickler workaround
def load_model_safely(path):
    import pickle
    import types
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy.random" and name == "__RandomState_ctor":
                return lambda *args: np.random.RandomState()
            return super().find_class(module, name)
    
    with open(path, 'rb') as f:
        return CustomUnpickler(f).load()

def expand_features(inputs):
    """
    Transform 10 basic skills into 470 features
    Example implementation - you'll need to customize this
    """
    import numpy as np
    
    # Start with original 10 features
    features = np.array(inputs)
    
    # Add polynomial features (example)
    poly_features = np.prod(np.array(inputs).reshape(-1, 1) * np.array(inputs))
    features = np.concatenate([features, poly_features.flatten()])
    
    # Add interaction terms (example)
    for i in range(len(inputs)):
        for j in range(i+1, len(inputs)):
            features = np.append(features, inputs[i] * inputs[j])
    
    # Pad with zeros if still needed (temporary fix)
    if len(features) < 470:
        features = np.pad(features, (0, 470 - len(features)))
    
    return features[:470]  # Ensure exactly 470 features
try:
    model = load_model_safely('final_model.sav')
except Exception as e:
    print(f"Critical Error: {str(e)}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Transform 10 inputs to 470 features
        expanded_features = expand_features(data['skills'])
        features = np.array(expanded_features).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"role": str(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)