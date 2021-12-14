# Imports
import cv2
import threading
import numpy as np
from mss import mss
# Tensorflow
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils

class CNN:
 
    # Peaked at 93% accuracy on 11 classes
 
    # Input Size = (500, 500, 3)
    def __init__(self, input_shape = (500, 500, 1), file_name = 'model'):
 
        # ==============Model Creation==============
        # Input Shape
        self.input_shape = input_shape
        # Model Save Name
        self.file_name = file_name
        # Initialize Model
        self.model = models.Sequential()
 
 
        # Shape = (500, 500, 3)
        # Conv2D #1
        self.model.add(layers.Conv2D(filters = 64, kernel_size = 17, strides = (3, 3), activation = None, input_shape = self.input_shape)) # was kernel = 16
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        # Shape = (162, 162, 64)
        # MaxPool2D #1
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))
        # Dropout
        self.model.add(layers.Dropout(0.5))  # This does not work if I put it before the max pool...
 
 
        # Shape = (81, 81, 64)
        # Conv2D #2
        self.model.add(layers.Conv2D(filters = 96, kernel_size = 9, strides=(3, 3), activation = None)) # was kernel = 32
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        # Shape = (25, 25, 96)
        # Padding                             (top, bottom), (left, right)
        self.model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
        # Shape = (26, 26, 96)
        self.model.add(layers.MaxPool2D(pool_size = (2, 2)))
        # Shape = (13, 13, 96)
        self.model.add(layers.Dropout(0.3))

        # Flatten
        self.model.add(layers.Flatten())
 
 
        # Shape = (16_224,)
        self.model.add(layers.Dense(units=1024, activation=None))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.Dropout(0.2))
 
 
        self.model.add(layers.Dense(units=512, activation='relu'))
 
        self.model.add(layers.Dense(units=11, activation='softmax'))
 
 
 
        # Model Compile
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.Adam(learning_rate=0.0005) # was 0.001
 
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics=['accuracy'],
        )


        # ==============Image Processing==============
        # BGR Separators for Red
        self.red_min = 180
        self.red_max = 255
        self.green_max = 100
        self.blue_max = 100


    def __str__(self):
        self.model.summary()
        return ''

    def save_model(self):
        self.model.save(filepath = f'.models/{self.file_name}')

    def load_model(self):
        self.model = models.load_model(filepath = f'.models/{self.file_name}')

    def predict_data_screenshot(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image2', image)
        cv2.waitKey(0)
        image = utils.img_to_array(image)
        image = tf.expand_dims(image, 0)
        predictions = self.model.predict(image)
        return predictions[0]

    def predict_raw_screenshot(self, image):
        # Process Screenshot
        image = self.process_raw_screenshot(image)
        cv2.imshow('image', image)
        cv2.waitKey(1)
        # Format to Array of Proper Dimensions
        image = utils.img_to_array(image)
        image = tf.expand_dims(image, 0) # Create a Batch
        # Predict
        predictions = self.model.predict(image)
        # Print and Return
        return predictions[0]

    # Check if Image needs to be Processed
    def process_raw_screenshot(self, image):
        # Drop Alpha Channel
        image = image[:, :, :3]
        # Filtered Red Locations
        image = cv2.inRange(image, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
        return image


class GatherData:

    # Gather Resolution = 250 x 250
    gather_resolution = 500

    def __init__(self, save_folder = '.data/Tracer/', resolution = (1920, 1080), time_between_frames = 100, start_file_number = 0, max_size = 1500):
        # Max Size
        self.max_size = max_size
        
        # Save Location
        self.save_folder = save_folder
        self.file_number = start_file_number

        # Resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]

        # Time Between Frames
        self.time_between_frames = time_between_frames

        # BGR Separators for Red
        self.red_min = 170
        self.red_max = 255
        self.green_max = 100
        self.blue_max = 100

        # Initialize screen capture to ammo location to middle 1/5th of screen
        self.monitor = {'top': round(self.y_res/2-(self.gather_resolution/2)), 'left': round(self.x_res/2-(self.gather_resolution/2)), 'width': self.gather_resolution, 'height': self.gather_resolution}

        # Initialize Capture Thread
        self.capture_thread = threading.Thread(target=self.capture_loop).start()

    def get_frame(self):
        with mss() as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.monitor))
        return frame

    def process_frame_for_save(self, image):
        # Drop Alpha Channel
        image = image[:, :, :3]
        # Filtered Red Locations
        image = cv2.inRange(image, (0, 0, self.red_min), (self.blue_max, self.green_max, self.red_max))
        # cv2.imshow('image', image)
        return image

    def save_frame(self, frame):
        # Save Frame
        cv2.imwrite(f'{self.save_folder}{self.file_number}.png', frame)
        self.file_number += 1

    # Main Data Capture Loop
    def capture_loop(self):
        while self.max_size>self.file_number:
            # Get Frame
            frame = self.get_frame()
            # Process Frame
            frame = self.process_frame_for_save(frame)
            # Save Frame
            self.save_frame(frame)
            # Wait
            cv2.waitKey(self.time_between_frames)

        # Break condition
        cv2.imshow('imagesdf', cv2.imread('test.png'))