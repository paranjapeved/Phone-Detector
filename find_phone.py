import cv2
import numpy as np
import os
from keras.models import model_from_json
import sys

def preprocess_images(images):
    return images / 255.0

if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Please enter path of test image as argument")
    else:

        path = sys.argv[1]
        print(path)
        test_img = cv2.imread(path,1)
        test_img = preprocess_images(test_img)
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")

        loaded_model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])

        prediction = loaded_model.predict(np.asarray([test_img]))
        os.system('clear')
        print(prediction[0][0],prediction[0][1])