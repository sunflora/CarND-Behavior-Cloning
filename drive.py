'''
    drive.py - The script to drive the car.     
'''

import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import cv2

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

DEBUG = False
DEBUG_IMAGES = False
STEERING_ADJUSTMENT_FACTOR = 10.
SMOOTH_RIDE_HISTORY_COUNT = 5
SMOOTH_RIDE_HISTORY_WEIGTH = - 0.3

steering_angles_history = []

def rgb_to_bgr(image):
    image.convert('RGB')
    open_cv_image = numpy.array(pil_image)
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

train_image_shape = (40, 160, 3)
def preprocess_image(image):
    global train_image_shape

    if DEBUG_IMAGES: print("-- Pre-processing image, resize into shape of ", train_image_shape)
    image = image[55:image.shape[0]-25, 0:image.shape[1], :]  
    image = cv2.resize(image, (train_image_shape[1], train_image_shape[0]), interpolation=cv2.INTER_AREA)

    if DEBUG_IMAGES: print("-- Pre-processing image, change image from RGB2HSV")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image

def get_smooth_angle(angle, history):
    sum = 0
    for h in history:
        sum = sum + h
    average = sum / len(history)
    if DEBUG: print("average: ", average)
    angle_smooth = angle * (1 - SMOOTH_RIDE_HISTORY_WEIGTH) + average * (SMOOTH_RIDE_HISTORY_WEIGTH)
    if DEBUG: print("angle_smooth: ", angle_smooth)
    return angle_smooth

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

indx = 0

@sio.on('telemetry')
def telemetry(sid, data):
    global SMOOTH_RIDE_HISTORY_COUNT
    global indx, steering_angles_history

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    if DEBUG_IMAGES:
        if (indx < 6): 
            filename = "output_images/" + str(indx) + ".jpg"
            indx = indx + 1
            image.save(filename, "JPEG")
    
    image_array = np.asarray(image)   
    
    # Added preprocessing to the image
    image_array = preprocess_image(image_array)

    if DEBUG_IMAGES:
        if (indx < 6):
            filename = "processed_images/" + str(indx) + ".jpg"
            indx = indx + 1
            cv2.imwrite(filename, image_array)

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    steering_angle = steering_angle / STEERING_ADJUSTMENT_FACTOR

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # throttle = 0.2
    
    if (len(steering_angles_history) == 0): 
        for i in range(0, SMOOTH_RIDE_HISTORY_COUNT):
            steering_angles_history.append(steering_angle)
    else:
        del steering_angles_history[0]
        steering_angles_history.append(steering_angle)
        if DEBUG: 
            for i in steering_angles_history:
                print(i)
        
    steering_angle_smooth = get_smooth_angle(steering_angle, steering_angles_history)
    steering_angle = steering_angle_smooth

    if float(speed) > 23:
        throttle = -0.1
    elif float(speed) > 20:
        throttle = 0.1
    elif float(speed) > 15:
        throttle = 0.15
    elif float(speed) > 10:
        throttle = 0.2
    else:
        throttle = 0.5

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
#        model = model_from_json(jfile.read())
        model = model_from_json(json.loads(jfile.read()))


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)