import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

class tumor:
    def __init__(self,filename):
        self.filename =filename


    def tumor_classify(self):

        model_path = "brain_tumor_model.h5"
        loaded_model = tf.keras.models.load_model(model_path)

        imagename = self.filename
        image = cv2.imread(imagename)

        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((224, 224))
        expand_input = np.expand_dims(resize_image,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255
        pred = loaded_model.predict(input_data)
        #result = pred.argmax()

        if pred >= 0.5:
            prediction = 'Tumor Present.'
            return [{"image": prediction}]
        elif pred < 0.5:
            prediction = 'Tumor Absent.'
            return [{"image": prediction}]
        else:
            return [{"ERROR": "Please upload another image!"}]

