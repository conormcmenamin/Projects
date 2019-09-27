import base64
import numpy as np
import io
import keras
from PIL import Image
from keras import backend as backend
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from flask import request
from flask import jsonify
from flask import Flask, render_template

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/')

def home():
    return render_template('layout.html')

@app.route('/pred',methods=["POST"])
def pred():
        message = request.json["image"]
        decoded = base64.b64decode(message)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(256,256,1))

        prediction = model.predict(processed_image).tolist()
        CATEGORIES = ["Angry","Fear", "Happiness","Neutral","Sadness","Surprise"]

        m = max(prediction)
        j = [ind for ind, val in enumerate(prediction) if val==m]

        response={
            'verdict': CATEGORIES[j]
        }

        return jsonify(response)
def get_model():
    global model
    model = load_model('myModel.h5')
    print('Model loaded')

def preprocess_image(image, target_size):
    if image.mode=='RGB':
        image = image.convert('L')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print('Loading neural network...')
get_model()
