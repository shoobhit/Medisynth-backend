from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import uuid
from model_inference import XrayModel

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load model
model = XrayModel('chexnet_custom_finetuned (1).pth')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the X-ray Classification API'})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            result = model.generate_report_and_cam(filepath)
            cam_filename = f"cam_{unique_filename}.png"
            cam_filepath = os.path.join(app.config['STATIC_FOLDER'], cam_filename)
            cv2.imwrite(cam_filepath, cv2.cvtColor(result['cam_image'], cv2.COLOR_RGB2BGR))

            report_filename = f"report_{unique_filename}.txt"
            report_filepath = os.path.join(app.config['STATIC_FOLDER'], report_filename)
            with open(report_filepath, 'w') as f:
                f.write(result['report'])

            return jsonify({
                'predictions': result['predictions'],
                'report': result['report'],
                'cam_url': f"/static/{cam_filename}",
                'report_url': f"/static/{report_filename}",
                'top_class': result['top_class']
            })
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/static/<filename>')
def serve_static(filename):
    return send_file(os.path.join(app.config['STATIC_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)















