from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the dehazing model
model = tf.keras.models.load_model("twoghaze1k.h5")

def process_image(image_data):
    """Process image data and return dehazed image"""
    # Convert base64 image data to PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.convert('RGB')
    
    # Preprocess image
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = (img / 127.5) - 1.0
    img = tf.image.resize(img, [256, 256])
    input_image = tf.expand_dims(img, 0)
    
    # Get prediction
    prediction = model(input_image, training=True)
    res = prediction[0] * 0.5 + 0.5
    
    # Convert result to base64
    res_image = Image.fromarray((res.numpy() * 255).astype(np.uint8))
    buffered = io.BytesIO()
    res_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

@app.route('/dehaze', methods=['POST'])
def dehaze():
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process image
        result_image = process_image(data['image'])
        
        return jsonify({
            'status': 'success',
            'dehazed_image': result_image
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
