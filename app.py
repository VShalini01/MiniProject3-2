import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
model = load_model("mini_project.keras")
IMG_SIZE = 240

# Check file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return "Not AI Generated" if prediction[0][0] > prediction[0][1] else "AI Generated"

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            file_url = url_for('static', filename='uploads/' + file.filename)

    return render_template('index.html', prediction=prediction, file_url=file_url)

if __name__ == '__main__':
    app.run(debug=True)
