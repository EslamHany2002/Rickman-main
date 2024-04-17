import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

LABELS = ['The image contains a brain tumor and the type is Glioma',
          'The image contains a brain tumor and the type is Meningioma',
          'The image does not contain a brain tumor.',
          'The image contains a brain tumor and the type is Pituitary']

model = load_model('D:\Rickman-main\Rickman-main\src\CNN\models\BrainTumor.h5')


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            input_img = preprocess_image(image)
            result = model.predict(input_img)
            predicted_class_index = np.argmax(result)
            predicted_class = LABELS[predicted_class_index]
            return jsonify({'result': predicted_class})
        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run()


#cd F:\Abdelrhman\Rickman-main\src\WebApp
#python app.py