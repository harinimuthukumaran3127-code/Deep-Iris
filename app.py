from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load trained model
model = load_model("eye_disease_model.h5")   # unga model file name same ah irukanum

# Class names
classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img_path = "static/" + img.filename
    img.save(img_path)

    # Image preprocess
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    pred = model.predict(img_array)
    result = classes[np.argmax(pred)]

    return render_template("index.html", prediction=result, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)

