import json
import numpy as np
import cv2,os
import glob
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout
from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
import re

# I
img_width = 300
img_height = 300

# Set directory path here
directory_path = "."
os.chdir(directory_path)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

batch_size = 32
epochs = 50
Y = None

# 
def objectpathtoX(obpath):
    X = np.array
    # image_path = "images"
    images_list = glob.glob(os.path.join(obpath,"*.png"))
    images_list.sort(key= lambda x:int(re.findall("(\d*)\.png",x)[0]))
    #reading all images as numpy arrays
    for image in images_list:
        X_image = cv2.imread(image)
        X = np.vstack(X, X_image)
    return X

def fetchX():
    os.chdir(os.path.join(directory_path,"images"))
    folders = os.listdir(".")
    finX = np.array
    for f in folders:
        if not os.path.isdir(f):
            print (folder)
            continue
        Xarr = objectpathtoX(f)
        print(Xarr)
        finX = np.vstack(finX,Xarr)
    return finX

#loading y labels and creating train-test split: json_file = file path of labels
def fetchY():
    Y = None
    with open('pose.json') as labels:
        Y = np.array(json.load(labels))
        Y.reshape(Y.shape[0],Y.shape[1],1,12)
    if Y is not None:
        return Y
    else:
        print ("Y is None")


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)

model = Sequential()
model.add(Conv2D(16, kernel_size = (8,8), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(7, (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(12))
sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss = losses.mean_squared_error, optimizer = 'sgd', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

predictions = model.predict(x_test, batch_size = batch_size)
print predictions
