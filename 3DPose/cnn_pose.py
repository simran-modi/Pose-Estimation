import json
import numpy as np
import cv2,os
import glob
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras import backend as K
import re

# Setting image size
img_width = 64
img_height = 64

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
    X = None
    # image_path = "images"
    images_list = glob.glob(os.path.join(obpath,"*.png"))
    images_list.sort(key= lambda x:int(re.findall("(\d*)\.png",x)[0]))
    #reading all images as numpy arrays
    for image in images_list:
        X_image = cv2.imread(image)
        if X is None:
            X = X_image.reshape(1,64,64,3)
        else:
            X = np.vstack((X, [X_image]))
    return X

def fetchX():
    os.chdir(os.path.join(directory_path,"images"))
    folders = os.listdir(".")
    #Uncomment for multiple objects
    #finX = None
    #Comment Xarr declaration for multiple objects (optional)
    Xarr = None
    for f in folders:
        if not os.path.isdir(f):
            print (f)
            continue
        Xarr = objectpathtoX(f)
        print(Xarr.shape)
    #Uncomment the following lines for pose estimation for multiple objects
#	if finX is None:
#	    finX = Xarr.reshape(1,64,64,3)
#	else:
#	    finX = np.vstack((finX, Xarr))
    #return finX
    return Xarr

#loading y labels and creating train-test split: json_file = file path of labels
def fetchY():
    Y = None
    with open(os.path.join(directory_path,'pose.json')) as labels:
        Y = np.array(json.load(labels))
        Y = Y.reshape(2562,-1)
    if Y is not None:
        return Y
    else:
        print ("Y is None")

X = fetchX()
Y = fetchY()
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

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
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss = losses.mean_squared_error, optimizer = 'sgd', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 2)

predictions = model.predict(x_test, batch_size = batch_size)
print(predictions)