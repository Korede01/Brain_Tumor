"""CNN Model Using Transfer Learning"""

#Import all Frameworks and Technologies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import warnings
warnings.filterwarnings('ignore')

import os
import math
import shutil

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

with zipfile.ZipFile('C:/Users/LENOVO/Documents/ML/Dataset/archive.zip', 'r') as zip_ref:
    zip_ref.extractall()

#Count the number of images in the respective classes 0 - Brain Tumour and 1 - Healthy

ROOT_DIR = 'C:/Users/LENOVO/Documents/Project/Brain Tumor/brain_tumor_dataset'

number_of_images = {}

for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))

"""Split Dataset
70% for Train Data
15% for Validation
15% for testing"""

def dataFolder(p, split):
    if not os.path.exists("./"+p):
        os.mkdir("./"+p)

        for dir in os.listdir(ROOT_DIR):
            os.makedirs("./"+p+"/"+dir)

            for img in np.random.choice(a= os.listdir(os.path.join(ROOT_DIR, dir)),size=(math.floor(split*number_of_images[dir])-5), replace=False):
                O = os.path.join(ROOT_DIR, dir, img) #path
                D = os.path.join("./"+p, dir)
                shutil.copy(O,D)
                os.remove(O)
    else:
        print("The folder exists")

dataFolder("train", 0.7) #Training Data Folder
dataFolder("val", 0.15) #Validation Data Folder
dataFolder('test', 0.15) #Testing Data Folder

#Building Our Model (CNN)

model = Sequential()
model.add(Conv2D(filters= 16, kernel_size=(3,3), activation='relu', input_shape=(224, 224, 3)))

model.add(Conv2D(filters= 36, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters= 64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters= 128, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

#Compile Model
model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

#Preparing our data using Generator

def preprocessingImages1(path):

    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input, horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')

    return image

path = "C:/Users/LENOVO/Documents/Project/Brain Tumor/train"
train_data = preprocessingImages1(path)

def preprocessingImages2(path):

    image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')

    return image

path = "C:/Users/LENOVO/Documents/Project/Brain Tumor/test"
test_data = preprocessingImages2(path)

path = "C:/Users/LENOVO/Documents/Project/Brain Tumor/val"
val_data = preprocessingImages2(path)


base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)

for layer in base_model.layers:
    layer.trainable = False

X = Flatten()(base_model.output)
x = Dense(units=1, activation="sigmoid")(X)

model = Model(base_model.input, x)

model.compile(optimizer='rmsprop', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])


hist = model.fit_generator(train_data, epochs=100, validation_data=val_data, verbose=1,)

model.save('model1.h5')  #Save Model

model = load_model("C:/Users/LENOVO/Documents/ML/Brain Tumor/Brain_Tumor/model1.h5")

acc = model.evaluate_generator(test_data)[1]

print(f"The accuracy is {acc*100} %")

path = "C:/Users/LENOVO/Documents/ML/Brain Tumor/Brain_Tumor/no/18 no.jpg"

img = load_img(path, target_size= (224, 224))

i = img_to_array(img)/255

plt.imshow(i)
plt.show()

input_arr = np.array([i])
pred = (model.predict(input_arr)[0][0] > 0.5).astype("int32")
#pred = np.argmax(model.predict(input_arr),axis=0)


if pred == 0:
    print("The MRI is not having a Tumor")
else:
    print("The MRI is having a Tumor")
