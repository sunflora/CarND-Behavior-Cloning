Introduction

What I enjoy the most of participating in the Udacity Self-driving Car Engineer Nanodegree 
Program is spending time implementation assignment projects.  You can see my previous projects and 
write-ups here and here.  This third project is called "Use Deep Learning to Clone 
Driving Behavior."  Udacity put together an awesome simulator that provides a game-like 
driving environment, where the driving behavior of the human player can be recorded and fed into
a deep learning network to learn how to drive.  After the network is properly trained, it will
have the ability to give instructions to the simulator on how to 'drive' the car autonomously.  

How does it work under the hood?

Just like a human player determines on how to drive the car in the simulator by viewing the road
conditions (for example, the road ahead is curving towards right or left) displayed on the 
computer screen, in our project, the neural network will take input of the series of images
made up of the game video.  Each image is passed through the deep learning network and a steering
angle is calculated based on the network and the weight of the network.  
 
In this project, our goal is to create a deep learning model using Keras, a high-level neural networks
library running on top of Tensorflow, also an open source library for machine learning algorithms.
The base of the data is the set of images taken from a central camera mounted in the car windshield 
simulator as well as steering angle associated with each image.  

Here is an example of an example image:

[IMAGE]

This project's learning from behavoir direction to self-driving car is very similar to that of 
comma.ai, a project whose "approach to Artificial Intelligence for self-driving cars is based 
on an agent that learns to clone driver behaviors and plans maneuvers by simulating future 
events in the road."  Comma.ai's approach can be seen here: https://arxiv.org/pdf/1608.01230v1.pdf

I have adopted Comma.ai's model presented on this codebase:
 [https://github.com/commaai/research/blob/master/train_steering_model.py] 

It is made up of three convolution layers, a flatten layer, two fully connected layer, and four
exponential linear units (ELUs) and two dropout layers as depicted below:

While these layers introduce nonlinearity into the model, data is normalized by the lamba layer
to [-0.5, 0.5].

[IMAGE]

I have kept most of the parameters the same as defined in Comma.ai's train_steering_model.py.
With one excption, the first dropout rate (p) was increased from 0.2 to 0.5.  This is due to
discovering exploding gradients as well as overfitting occurs quickly with the number of epochs.
Another hyper-parameter that was set for this reason is the learning rate.  It was set from 0.001
the default value to 0.0001.  

Inspired by the transfer learning, I have designed the model.py to be able to train an existing
model.  The non-tabula rasa approach saved me some time working on this project and here is my
training log.  As you can see in this instance, I ran the model.py five times, each time with
epoch value of 25, 10, 5, 5, and 5.  Note that I have multiply the steering value by 10 and
therefore you will see the loss number much larger (10x10 times due to mean square error) than
without the modification.  In drive.py, the prediction values were divided by 10 prior to sending
to the simulator.

Buidling model from ground up...
Creating model ...
Creating model completed.
Training model ...
Epoch 1/25
38675/38675 [==============================] - 87s - loss: 2.8625 - val_loss: 2.4416
Epoch 2/25
38675/38675 [==============================] - 85s - loss: 2.3935 - val_loss: 2.2758
Epoch 3/25
38675/38675 [==============================] - 81s - loss: 2.2585 - val_loss: 2.1973
Epoch 4/25
38675/38675 [==============================] - 111s - loss: 2.1424 - val_loss: 2.0921
Epoch 5/25
38675/38675 [==============================] - 96s - loss: 2.0424 - val_loss: 1.9404
Epoch 6/25
38675/38675 [==============================] - 102s - loss: 1.9349 - val_loss: 1.9258
Epoch 7/25
38675/38675 [==============================] - 98s - loss: 1.8483 - val_loss: 1.8116
Epoch 8/25
38675/38675 [==============================] - 90s - loss: 1.7748 - val_loss: 1.7388
Epoch 9/25
38675/38675 [==============================] - 100s - loss: 1.7024 - val_loss: 1.7172
Epoch 10/25
38675/38675 [==============================] - 92s - loss: 1.6391 - val_loss: 1.6815
Epoch 11/25
38675/38675 [==============================] - 85s - loss: 1.5941 - val_loss: 1.6796
Epoch 12/25
38675/38675 [==============================] - 79s - loss: 1.5590 - val_loss: 1.6712
Epoch 13/25
38675/38675 [==============================] - 84s - loss: 1.5177 - val_loss: 1.5791
Epoch 14/25
38675/38675 [==============================] - 84s - loss: 1.4716 - val_loss: 1.5133
Epoch 15/25
38675/38675 [==============================] - 81s - loss: 1.4451 - val_loss: 1.5339
Epoch 16/25
38675/38675 [==============================] - 81s - loss: 1.4087 - val_loss: 1.5014
Epoch 17/25
38675/38675 [==============================] - 80s - loss: 1.3922 - val_loss: 1.4751
Epoch 18/25
38675/38675 [==============================] - 81s - loss: 1.3586 - val_loss: 1.4579
Epoch 19/25
38675/38675 [==============================] - 81s - loss: 1.3375 - val_loss: 1.4751
Epoch 20/25
38675/38675 [==============================] - 81s - loss: 1.3157 - val_loss: 1.4828
Epoch 21/25
38675/38675 [==============================] - 81s - loss: 1.2806 - val_loss: 1.5641
Epoch 22/25
38675/38675 [==============================] - 86s - loss: 1.2710 - val_loss: 1.4850
Epoch 23/25
38675/38675 [==============================] - 86s - loss: 1.2611 - val_loss: 1.5739
Epoch 24/25
38675/38675 [==============================] - 89s - loss: 1.2445 - val_loss: 1.5939
Epoch 25/25
38675/38675 [==============================] - 84s - loss: 1.2316 - val_loss: 1.5823

Learning from existing model:  my_model.h5
Training model ...
Epoch 1/10
38675/38675 [==============================] - 93s - loss: 1.2327 - val_loss: 1.4614
Epoch 2/10
38675/38675 [==============================] - 80s - loss: 1.2145 - val_loss: 1.3982
Epoch 3/10
38675/38675 [==============================] - 78s - loss: 1.1983 - val_loss: 1.4575
Epoch 4/10
38675/38675 [==============================] - 78s - loss: 1.1808 - val_loss: 1.4437
Epoch 5/10
38675/38675 [==============================] - 80s - loss: 1.1704 - val_loss: 1.3959
Epoch 6/10
38675/38675 [==============================] - 79s - loss: 1.1609 - val_loss: 1.3531
Epoch 7/10
38675/38675 [==============================] - 79s - loss: 1.1410 - val_loss: 1.3253
Epoch 8/10
38675/38675 [==============================] - 79s - loss: 1.1391 - val_loss: 1.3795
Epoch 9/10
38675/38675 [==============================] - 81s - loss: 1.1204 - val_loss: 1.3755
Epoch 10/10
38675/38675 [==============================] - 77s - loss: 1.1102 - val_loss: 1.4245
Training model completed.

Learning from existing model:  my_model.h5
Training model ...
Epoch 1/5
38675/38675 [==============================] - 84s - loss: 1.0624 - val_loss: 1.2328
Epoch 2/5
38675/38675 [==============================] - 83s - loss: 1.0634 - val_loss: 1.2241
Epoch 3/5
38675/38675 [==============================] - 83s - loss: 1.0502 - val_loss: 1.2047
Epoch 4/5
38675/38675 [==============================] - 80s - loss: 1.0330 - val_loss: 1.1476
Epoch 5/5
38675/38675 [==============================] - 82s - loss: 1.0252 - val_loss: 1.1748
Training model completed.

Learning from existing model:  my_model.h5
Training model ...
Epoch 1/5
38675/38675 [==============================] - 109s - loss: 1.0350 - val_loss: 1.2697
Epoch 2/5
38675/38675 [==============================] - 111s - loss: 1.0238 - val_loss: 1.3697
Epoch 3/5
38675/38675 [==============================] - 93s - loss: 1.0151 - val_loss: 1.3176
Epoch 4/5
38675/38675 [==============================] - 94s - loss: 1.0035 - val_loss: 1.2951
Epoch 5/5
38675/38675 [==============================] - 83s - loss: 0.9887 - val_loss: 1.2426
Training model completed.

Learning from existing model:  my_model.h5
Training model ...
Epoch 1/5
38675/38675 [==============================] - 137s - loss: 0.9852 - val_loss: 1.3177
Epoch 2/5
38675/38675 [==============================] - 173s - loss: 0.9707 - val_loss: 1.3585
Epoch 3/5
38675/38675 [==============================] - 164s - loss: 0.9679 - val_loss: 1.3934
Epoch 4/5
38675/38675 [==============================] - 159s - loss: 0.9650 - val_loss: 1.3304
Epoch 5/5
38675/38675 [==============================] - 159s - loss: 0.9592 - val_loss: 1.3207

Aside from the learning model, the most important aspect of this project is actually the data itself.
Here are some processes taken regarding data:

1. Data gathering: While it is fun to play with the simulator, the method for steering maneuvers 
via holding down the left-mouse button was not very user friendly (for some reason the keyboard
entry does not work on my MacBook Pro.)  Luckily, Udacity provides a set of data [link] and I end
up using it for the project instead of my own generated ones.  Here are a step of steps I took to
pre-processing the data:

2. Data balancing: 
    a. There are more left-turns on the driving course than right-turns.  The way to 
    counter this imbalance is to transform and add vertically flipped images to the dataset.  
    b. There are a lot of data with steering angle 0 in the database.  Instead of removing
    some of these data and then figure out other ways to argument the data, I decided to insert
    data with steering angle > 0.01 or < -0.01 into the dataset.

3. Data argumentation:  
    a. In the database provided by Udacity, besides the images taken from the centrally mounted
    camera, there are also images taken from the left- and right-mounted camera.  While the drive.py
    takes only consideration of images from the central camera.  The images taken by the side cameras
    provides valuable information and with little manupilation they can also be used for data 
    argumentation. 

4. Image pre-processing:
    a. Determine the Region of Interest (ROI).  Only the middle section of the images with road 
    information are kept as ROI.  The image size is reduced from 320x160 to 320x80.
    b. The images are downsampled by 2x2. The image size is further reduced to 160x40.
    c. The images are converted to HSV channel.

[Here are examples of these images]

5. Ampifying the steering values:
    The steering values are very small and their differences to 0 is very minimal.  I decided to
    multiply them by 10 to increase the distance between turning and going straight.  This essentially
    increase the steering output from the network 10 times.  Therefore, it is important to divide
    the model prediction by 10 before feeding it to the simulator.

6. Finally, 10% of the training dataset were splitted out to use as the validation dataset.


The result of track one can be seen here:
[Link - Track #1]

Variable Throttle Values on drive.py

Since the model is trained on Track 1 images, Track 2 is the ultimate test for generalization. 
My initial Track 2's steering performance was adequate but the car fell back down the road when 
trying to climb up.  Track 1 is mostly flat so any throttle value would likely be enough 
to propel forward (I chose a constant throttle = 0.15 to combat the wobbliness from 
Udacity's default 0.2.)  In order to help the car successfully climbing up the mountain road 
(as it already sucessfully detect reasonable steering angle), I put in variable throttle value 
determined based on the current speed.  Here is my code for variable throttle setting:

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

As you can see, I also added a small negative value if the car is going too much of high speed.
This is useful to prevent the car from hitting the shoulder while going down hill and needing to 
make a sharp turn.

Reverse Average of the Steering Angles

While the car can be driven autonomously, I was still not satisfied with the fact that it is 
wobbly too much.  This unsteady movement will surely give the passenager a headache.  I experimented
with a few technics to let the car drive more steady.  I decided to emply the Reverse Averaging
technic that tone down the speed based on average of speed of past several frames.  Here is my
function to calculate the smoother steering angle:

def get_smooth_angle(angle, history):
    sum = 0
    for h in history:
        sum = sum + h
    average = sum / len(history)
    if DEBUG: print("average: ", average)
    angle_smooth = angle * (1 - SMOOTH_RIDE_HISTORY_WEIGTH) + average * (SMOOTH_RIDE_HISTORY_WEIGTH)
    if DEBUG: print("angle_smooth: ", angle_smooth)
    return angle_smooth

Note that the SMOOTH_RIDE_HISTORY_WEIGTH is set to a small negative number (therefore the term 
Reverse Average).  I have setting a value 0.3 for a count of 5 of the past frames.  

The result can be seen here:

[link1]
[link2]
