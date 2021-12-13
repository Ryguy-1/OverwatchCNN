from trainer import GatherData
from trainer import CNN
# Numpy
import numpy as np
# Tensorflow
import tensorflow as tf
import tensorflow.keras.utils as utils
# MSS
from mss import mss
# OpenCv
import cv2
# Time
import time


# Set Seed Variable and Tensorflow / Numpy Seeds
seed = 42
tf.random.set_seed(seed)
np.random.seed = seed

# Print Cuda Enabled or Not
print(f'CUDA Enabled: {tf.test.is_built_with_cuda}')

# Hyperparameters
batch_size = 32
epochs = 17

def main():
    # Gather Data
    # trainer = GatherData(save_folder='.data2/Hanzo/', resolution=(1920, 1080), max_size=1500)

    # Model Initialization
    cnn = CNN()
    print(cnn)

    # Train Model
    train_model(cnn)

    # Run Model
    # run_model(cnn)

    # print('5 seconds left')
    # time.sleep(5)
    # save_screenshot(get_frame(), 'test.png')
    # probability_arr = cnn.predict_raw_screenshot(cv2.imread('test.png'))
    # print(probability_arr)
    
    # print(cnn.predict_data_screenshot(cv2.imread('.data/Tracer/3600.png')))
    
        
# ==============Use Model==============
threshold_confidence = 0.5
def run_model(cnn = None):
    # Load Model
    cnn.load_model()
    # Prediction Loop
    while True:
        prediction_arr = cnn.predict_raw_screenshot(get_frame())
        if prediction_arr[0]>threshold_confidence:
            print("Ashe"); print(prediction_arr)
        elif prediction_arr[1]>threshold_confidence:
            print("Tracer"); print(prediction_arr)


# ==============Train Model==============
def train_model(cnn = None):
    # Train Model
    train = utils.image_dataset_from_directory(
        directory = '.data2/', 
        label_mode = 'categorical', # categorical = one hot encoding for labels (works for categorical_crossentropy loss function), int = regular ints
        batch_size = batch_size, 
        image_size = (500, 500),
        shuffle = True,
        seed = seed,
        validation_split = 0.3,
        subset = 'training',
        color_mode = 'grayscale',
    )

    test = utils.image_dataset_from_directory(
        directory = '.data2/',
        label_mode = 'categorical',
        batch_size = batch_size,
        image_size = (500, 500),
        shuffle = True,
        seed = seed,
        validation_split = 0.3,
        subset = 'validation',
        color_mode = 'grayscale',
    )

    

    cnn.model.fit(
        x = train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data = test,
        shuffle = True,
        validation_batch_size=128
    )

    cnn.save_model()


# ==============Get Screenshot==============
y_res = 1080; x_res = 1920
gather_resolution = 500
def get_frame():
    with mss() as sct:
        # Get Screen Frame
        frame = np.array(sct.grab({'top': round(y_res/2-(gather_resolution/2)), 'left': round(x_res/2-(gather_resolution/2)), 'width': gather_resolution, 'height': gather_resolution}))
    return frame

def save_screenshot(image, name):
    cv2.imwrite(name, image)

if __name__ == "__main__":
    main()