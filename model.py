''' 
    model.py - The script used to create and train the model.
'''

import cv2
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Lambda, ELU, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

DEBUG = False
DEBUG_IMAGES = False
LOG_DIRECTORY = "dataUdacity/"
LOG_FILENAME = "driving_log.csv"
LOG_FULL_PATH = os.path.join(LOG_DIRECTORY, LOG_FILENAME)
SAVE_DIRECTORY = "saved_images/"
STEERING_ADJUSTMENT_FACTOR = 10.
RIGHT_IMAGE_STEERING_ADJUST = -0.25
LEFT_IMAGE_STEERING_ADJUST = 0.25
TURNING_INCREASE_FACTOR = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCH = 5

X_train = []
y_train = []
X_val = []
y_val = []
model = False
train_image_shape = (40, 160, 3)
my_model = "my_model.h5"

'''
    retrieve_image: reading and return the image object speicified by the filename
'''
def retrieve_image(filename):
    filename = filename.strip()
    filename = os.path.join(LOG_DIRECTORY, filename) 

    if DEBUG: print("-- Retrieving image: ", filename)
    image = cv2.imread(filename)
    return image

'''
    save_image: saving image object to the file system.
    This function is used  the result of the processing images.
'''

def save_image(image, filename):
    if DEBUG_IMAGES: print("-- Saving images into the directory of ", SAVE_DIRECTORY)
    filename = os.path.join(SAVE_DIRECTORY, filename)
    if DEBUG_IMAGES: print("-- Filename: ", filename)
    cv2.imwrite(filename, image)

'''
    preprocess_image: getting image ready for training.
    steps:
        1. Elimiate the irrevalent portions of the image, just keep the section with road and curve:
            - 55 pixels on top 
            - 25 pixels on the bottom
           The image size is now 320x80
        2. Downsize the image from 160x40.  This speeds up the training speed by a significant amount.
        3. Convert image into HSV space.
'''
def preprocess_image(image):
    global train_image_shape

    if DEBUG_IMAGES: print("-- Pre-processing image, resize into shape of ", train_image_shape)
    image = image[55:image.shape[0]-25, 0:image.shape[1], :]  
    image = cv2.resize(image, (train_image_shape[1], train_image_shape[0]), interpolation=cv2.INTER_AREA)

    if DEBUG_IMAGES: print("-- Pre-processing image, change image from BGR2HSV")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

'''
    get_flipped_image: function to flip image vertically so pixels on the left are now placed on the right.
'''    

def get_flipped_image(image, steering):
    flipped_image = cv2.flip(image, 1)
    flipped_steering = steering * -1.
    return flipped_image, flipped_steering

'''
    def create_validation_data(ratio): split ratio portion of the training data into validation data 
'''
def create_validation_data(ratio):
    global X_train, y_train, X_val, y_val
    
    if DEBUG: print("-- Creating validation data with the ratio of: ", ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio) 
    if DEBUG:
        print("-- X_train sizes: ", X_train.shape)
        print("-- y_train sizes: ", y_train.shape)  
        print("-- X_train sizes: ", X_train.shape)
        print("-- y_train sizes: ", y_train.shape)  
        
'''
    process_data: prepping the data prior to training.
    steps:
        1. retriving images from the file System
        2. processing images: cut out non-ROI(Region of Interest), down-sample, convert into HSV 
        3. argumenting data so the data now contains these groups:
            a. center images - images taken from the center camera
            b. flipped center images: aims to balance the right and left turn
            c. right images - images taken from the right camera
            d. left images - images taken from the left camera
            e. right turn and left turn images
'''
def process_data():
    global DEBUG, DEBUG_IMAGES
    global X_train, y_train
    
    print("Retrieving dataset ...  from:", LOG_FULL_PATH)
    driving_data = pd.read_csv(LOG_FULL_PATH, header=0, index_col = False)
    print("Retrieving dataset completed. The retrieved size of dataset is ",len(driving_data))

    print("Processing images ... ")
    files = driving_data['center']
    files_right = driving_data['right']
    files_left = driving_data['left']
    steerings = driving_data['steering']
    count_of_turning_images = 0 
    for f, fr, fl, s in zip(files, files_right, files_left, steerings):
        if DEBUG: 
            print("center file: ", f)
            print("right file: ", fr)
            print("left file: ", fl)

        image = retrieve_image(f)
        image = preprocess_image(image)
        if DEBUG:
            print("center image shape: ", image.shape)
        if DEBUG_IMAGES: 
            save_image(image, f)
        X_train.append(image)
        s_adj = s * STEERING_ADJUSTMENT_FACTOR
        y_train.append(s_adj)

        if DEBUG: print("Data argumentation: flipping central images")
        flipped_central, flipped_steering = get_flipped_image(image, s_adj)
        X_train.append(flipped_central)
        y_train.append(flipped_steering)

        if DEBUG: print("Data argumentation: using right images")
        image_right = retrieve_image(fr)
        if DEBUG: print("right image shape: ", image_right.shape)
        image_right = preprocess_image(image_right)
        s_right_adj = (s + RIGHT_IMAGE_STEERING_ADJUST) * STEERING_ADJUSTMENT_FACTOR 
        X_train.append(image_right)
        y_train.append(s_right_adj)

        if DEBUG: print("Data argumentation: using left images")
        image_left = retrieve_image(fl)
        if DEBUG: print("left image shape: ", image_left.shape)
        image_left = preprocess_image(image_left)
        s_left_adj = (s + LEFT_IMAGE_STEERING_ADJUST) * STEERING_ADJUSTMENT_FACTOR 
        X_train.append(image_left)
        y_train.append(s_left_adj)

        if TURNING_INCREASE_FACTOR > 0: 
            print("Data balancing: adding more turning data")    
            if ((s > 0.01) or (s < -0.01)):  #Both Right term and left terms are added
                count_of_turning_images = count_of_turning_images + 1
                for i in range(0, TURNING_INCREASE_FACTOR):
                    X_train.append(image)
                    y_train.append(s_adj)
                    X_train.append(image_right)
                    y_train.append(s_right_adj)
                    X_train.append(image_left)
                    y_train.append(s_left_adj)
        if DEBUG:
            if ((s_adj > 1 * STEERING_ADJUSTMENT_FACTOR) or (s_adj < -1 * STEERING_ADJUSTMENT_FACTOR)):
                print("------------  ALERT  ------- ", f ," s_adj over: ",  s_adj)
            if ((s_right_adj > 1 * STEERING_ADJUSTMENT_FACTOR) or (s_right_adj < -1 * STEERING_ADJUSTMENT_FACTOR)):
                print("------------  ALERT  ------- ", fr ," s_adj over: ",  s_right_adj)
            if ((s_left_adj > 1 * STEERING_ADJUSTMENT_FACTOR) or (s_left_adj < -1 * STEERING_ADJUSTMENT_FACTOR)):
                print("------------  ALERT  ------- ", fl," s_adj over: ",  s_left_adj)

    print("Count of Turning Images: " , count_of_turning_images)
    X_train = np.asarray(X_train) 
    y_train = np.asarray(y_train)
    print("Processing images completed.") 
    print("X_train sizes: ", X_train.shape)
    print("y_train sizes: ", y_train.shape)  

    print("Creating valudation dataset ...")   
    create_validation_data(0.1)
    print("X_train sizes: ", X_train.shape)
    print("y_train sizes: ", y_train.shape)  
    print("X_val sizes: ", X_val.shape)
    print("y_val sizes: ", y_val.shape)  
    
    print("Creating valudation dataset completed.")   

'''
    get_sample_data: this is used for initial experimenting with data 
'''
def get_sample_data():
    sample_data_filenames = ["IMG/center_2016_12_01_13_39_28_024.jpg","IMG/center_2016_12_01_13_40_08_646.jpg", "IMG/center_2016_12_01_13_30_48_287.jpg"]
    sample_steering_values = [-0.9426954, 0.5784606, 0]


'''
    create_model: creates the model to train the data
    The model is built via Keras and is based on comma.ai 
    https://github.com/commaai/research/blob/master/train_steering_model.py
'''
def create_model():
    global LEARNING_RATE
    global train_image_shape

    print("Creating model ...")
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=train_image_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())

#    model.add(Dropout(.2))
    model.add(Dropout(.5))

    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=adam,  loss="mse")
    print("Creating model completed.")

    return model


'''
    init_model: this function determines whether to train on an existing model or creating a new one.abs
    Here are the options:
        1. training from existing .h5 model
        2. training from existing .json/.h5 model
        3. creating a new model
'''
def init_model():
    global model

    my_model = "my_model.h5"
    if Path(my_model).is_file():
        model = load_model(my_model)
        print("Learning from existing model: ", my_model)

    elif Path("model.json").is_file():
        with open("model.json", 'r') as jfile:
            model = model_from_json(json.loads(jfile.read()))
        model.compile("adam", "mse")
        model.load_weights("model.h5")
        print("Learning from existing model: model.h5")

    else:
        print("Buidling model from ground up...")
        model = create_model()

'''
    data_generator: generating data used in keras model.fit_generator function
'''
def data_generator(X, Y):
    while 1:
        for x, y in zip(X, Y):
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = np.array([[y]])
            yield(x, y)

'''
    batch_data_generator: generating batch data used to fit model
'''
def batch_data_generator(X, Y, batch_size):
    while 1:
        count = 0
        features = []
        target = []
        for x, y in zip(X,Y):
            if count < batch_size:
                features.append(x)
                target.append(y)
                count = count + 1
            else:
                features.append(x)
                target.append(y)
                count = 0
                yield(np.array(features), np.array(target))
                features = []
                target = []
'''
    train_model: keras model.fit_generator is used to train the model
'''
def train_model():
    global X_train, y_train, X_val, y_val, BATCH_SIZE

    # samples_per_epoch=(BATCH_SIZE * np.ceil(X_train.shape[0]/BATCH_SIZE))
    # print("samples_per_epoch is: ", samples_per_epoch)

    samples_per_epoch = 38675

    print("Training model ...")
    model.fit_generator(batch_data_generator(X_train, y_train, BATCH_SIZE), 
                        samples_per_epoch=samples_per_epoch, nb_epoch=EPOCH, 
                        validation_data=batch_data_generator(X_val, y_val, BATCH_SIZE), 
                        nb_val_samples = X_val.shape[0])
    print("Training model completed.")

'''
    save_result: the model info (including weights, etc) is saved for future training or auto-driving
'''
def save_result():
    global model

    json_file = "model.json"
    if Path(json_file).is_file():
        os.remove(json_file)
    with open(json_file, 'w') as f:
        json.dump(model.to_json(), f)

    weights_file = "model.h5"
    if Path(weights_file).is_file():
        os.remove(weights_file)
    model.save_weights("model.h5")

    if Path(my_model).is_file():
        os.remove(my_model)

    model.save(my_model)


if __name__ == '__main__':
    process_data()
    init_model()
    train_model()
    save_result()

