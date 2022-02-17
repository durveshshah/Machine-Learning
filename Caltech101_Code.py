
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plotAxis
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pathlib
import time
from google.colab import drive

#=================Variable declaration=====================
drive.mount('/content/drive')
datasetPath = 'drive/MyDrive/MLProject/Caltech101'
#datasetPath = 'Caltech101'
totalClasses = []
trainData = []
testData = []
trainArray = []
testArray = []
trainingPath = 'TrainingData101'
testingPath = 'TestingData101'
imageHeight,imageWeight = 227,227
batchSize = 8
numEpochs = 50
#================== Processing Dataset =======================#

dataPath = os.getcwd()
dirLocation =pathlib.Path(datasetPath)
dataDirectory=os.listdir(dirLocation) # getting all different classes from dataset.
for classes in dataDirectory:
    totalClasses.append(classes)
classLen = len(dataDirectory)

#========== Getting total number of images from dataset=====
img = list(dirLocation.glob('*/*.jpg'))
totalImgs = len(img)

#========= Printing Dataset details=========================
print("Caltech-101 Dataset")
print('--------------------------------------------')
print('Dataset Name                               :', 'Caltech101 Dataset')
print('Total Number of classes in Caltech-101      :', classLen)
print('Total Number of Images in Caltech-256       :', totalImgs)

if ((os.path.isdir('TrainingData101') or os.path.isdir('TestingData101')) != True ):
 os.mkdir(os.path.join(dataPath, "TrainingData101"))
 os.mkdir(os.path.join(dataPath, "TestingData101"))
 for className in totalClasses:
    data = os.listdir(datasetPath + "/" + className + "/")
    x = data[:30]  # selecting random 30 images from each class for training
    y = data[30:]  # rest images used for testing
    trainArray.append([className, x])
    testArray.append([className, y])


 def getData(data_dir, newDataLocation,dataset):
   for data in dataset:
       if data[0] == 'BACKGROUND_Google':
           continue
       className = data[0]
       imageArray = data[1]
       finalPath = os.path.join(newDataLocation, className)
       os.mkdir(finalPath)
       for image in imageArray:
           imgPath = data_dir + '/' + className + '/' + image
           imageName = cv2.imread(imgPath, 0)
           cv2.imwrite(os.path.join(finalPath, image), imageName)
 newTrainingPath = dataPath + '//'+trainingPath
 newTestingPath  = dataPath + '//'+testingPath
 getData(datasetPath, newTrainingPath, trainArray)
 getData(datasetPath, newTestingPath, testArray)

#generate training and testing data from directory.
def generateTrainTestData(trainingPath,testingPath,imageHeight,imageWeight,batchSize):
    trainingDataset = tf.keras.preprocessing.image_dataset_from_directory(trainingPath, validation_split=0.1,
                                                                          subset="training",
                                                                          seed=123, label_mode='categorical',
                                                                          image_size=(imageHeight, imageWeight),
                                                                          batch_size=batchSize)

    validationDataset = tf.keras.preprocessing.image_dataset_from_directory(testingPath, validation_split=0.1,
                                                                            subset="validation",
                                                                            seed=123, label_mode='categorical',
                                                                            image_size=(imageHeight, imageWeight),
                                                                            batch_size=batchSize)
    return trainingDataset, validationDataset


trainingDataset, validationDataset = generateTrainTestData(trainingPath,testingPath,imageHeight,imageWeight,batchSize)


#Feature extraction
valDataset = validationDataset.skip(tf.data.experimental.cardinality(validationDataset) // 5)
trainingDataset = trainingDataset.prefetch(buffer_size=tf.data.AUTOTUNE)
valDataset = valDataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#Building Model
def createResnet101Model():
    pretrainedModel = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(imageHeight, imageWeight, 3),
                                                 pooling='max',
                                                 classifier_activation='softmax', classes=classLen)

    pretrainedModel.trainable = False
    resnetModel = Sequential()
    resnetModel.add(pretrainedModel)
    resnetModel.add(Flatten())
    resnetModel.add(layers.Dense(2048, activation=('relu')))
    resnetModel.add(layers.Dropout(.4))
    resnetModel.add(layers.Dense(1024, activation=('relu')))
    resnetModel.add(layers.Dropout(.3))  # Adding dropout layer
    resnetModel.add(layers.Dense(512, activation=('relu'), name="Deep_feature"))
    resnetModel.add(layers.Dropout(.2))
    resnetModel.add(layers.Dense(101, activation=('softmax')))
    resnetModel.summary()
    return resnetModel

resnetModel = createResnet101Model()
resnetModel.compile(optimizer=Adam(learning_rate=0.000055),loss='categorical_crossentropy',metrics=['accuracy'])
finalResult = resnetModel.fit(trainingDataset,validation_data=valDataset,epochs=numEpochs)

#=======================================================================================================
print("Plotting Accuracy graph for Caltech-101 Dataset :")
figure = plotAxis.gcf()
plotAxis.plot(finalResult.history['accuracy'],label='Training Accuracy')
plotAxis.plot(finalResult.history['val_accuracy'],label='Validation Accuracy')
plotAxis.legend()
plotAxis.xlabel('Epochs')
plotAxis.ylabel('Accuracy')
plotAxis.axis(ymin=0.1,ymax=1)
plotAxis.grid()
plotAxis.title('Caltech101 Accuracy')
plotAxis.show()

print("Resnet101 model for Caltech-101 Dataset executed successfully...")