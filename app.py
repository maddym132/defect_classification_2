import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
#from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.utils import load_img, img_to_array
#from utils import load_weights
#from tensorflow.keras.models import Sequential, load_model
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from datetime import datetime
import base64
import io


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'uploads'
model = torch.load('welding_torch_model.hdf5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict_image(image):
        
    log_msg('Predicting image')
    try:
        if image.mode != "RGB":
            log_msg("Converting to RGB")
            image = image.convert("RGB")

        preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        index = output.data.cpu().numpy().argmax()
        probability = torch.nn.functional.softmax(output[0], dim=0).data.cpu().numpy().max()
        return index, probability
    except Exception as e:
        log_msg(str(e))
        return 'Error: Could not preprocess image for prediction. ' + str(e)
    

# Helper to predict an image encoded as base64
def predict_image_base64(encoded_image):
    if encoded_image.startswith('b\''):
        encoded_image = encoded_image[2:-1]

    decoded_img = base64.b64decode(encoded_image.encode('utf-8'))
    img_buffer  = io.BytesIO(decoded_img)

    image = Image.open(io.BytesIO(decoded_img))
    return predict_image(image)

def log_msg(msg):
    print("{}: {}".format(datetime.now(),msg))

def predict_url(imageUrl):
    log_msg("Predicting from url: " +imageUrl)
    urllib.request.urlretrieve(imageUrl, filename="tmp.jpg")
    input_image = Image.open("tmp.jpg")
    return predict_image(input_image)



app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img = Image.open(file_path)
            output, _ = predict_image(img)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)
