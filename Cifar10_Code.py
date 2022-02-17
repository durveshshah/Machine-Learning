import tensorflow.keras as keras
import tensorflow.compat.v1 as tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plotAxis




print("Running Cifar10 Dataset :")
print('--------------------------------------------------------------------------------------')
print('Dataset Name                               :', 'Cifar10 Dataset')
print('--------------------------------------------------------------------------------------')

#Processing training and testing Data
def processCifarData(X, Y,X1,Y1):
    x_train = X / 255
    x_test = X1 / 255
    y_train = keras.utils.to_categorical(Y)
    y_test = keras.utils.to_categorical(Y1)
    return x_train, x_test,y_train,y_test

numEpochs = 15
learning_rate=0.000057
#============================Loading Cifar10 Data===================
(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()
x_train, x_test,y_train,y_test = processCifarData(x_train,y_train,x_test,y_test)

def getImages(image, label):
  img = tensorflow.image.resize(image, (227, 227))
  return img, label
#Combining Training and Testing Data
def dataConcatenation(x_train, x_test,y_train,y_test) :
    trainingData = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train))
    trainingData = (trainingData.map(getImages).batch(batch_size=32, drop_remainder=True))  
    testingData = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test))
    testingData = (testingData.map(getImages).batch(batch_size=32, drop_remainder=True))
    return trainingData, testingData

trainingData, testingData = dataConcatenation(x_train, x_test,y_train,y_test)

print("Data Processing is done......................................................")

def createResnetModel ():
    #Initializing Resnet101 model
    pretrainedModel = keras.applications.ResNet101(include_top=False,
                                              weights='imagenet',
                                              input_tensor=keras.Input(shape=(227, 227, 3)))

    for resnetLayer in pretrainedModel.layers:
        resnetLayer.trainable = True

    #Adding Layers in model
    resnetModel = Sequential()
    resnetModel.add(pretrainedModel)
    resnetModel.add(Flatten())
    resnetModel.add(layers.Dense(2048, activation=('relu')))
    resnetModel.add(layers.Dropout(.4)) # Adding dropout layer
    resnetModel.add(layers.Dense(1024, activation=('relu')))
    resnetModel.add(layers.Dropout(.3))  # Adding dropout layer
    resnetModel.add(layers.Dense(512, activation=('relu'), name="Deep_feature"))
    resnetModel.add(layers.Dropout(.2))  # Adding dropout layer
    resnetModel.add(layers.Dense(10, activation=('softmax')))
    resnetModel.summary()
    return resnetModel

resnetModel = createResnetModel()

#Compiling the model...
resnetModel.compile(optimizer=Adam(learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
finalResult = resnetModel.fit(trainingData, epochs=numEpochs,verbose=1, validation_data=(testingData))

#=======================================================================================================
print("Plotting Accuracy graph for Cifar10 Dataset :")
figure = plotAxis.gcf()
plotAxis.plot(finalResult.history['accuracy'],label='Training Accuracy')
plotAxis.plot(finalResult.history['val_accuracy'],label='Validation Accuracy')
plotAxis.legend()
plotAxis.xlabel('Epochs')
plotAxis.ylabel('Accuracy')
plotAxis.axis(ymin=0.1,ymax=1)
plotAxis.grid()
plotAxis.title('Cifar-10 accuracy')
plotAxis.show()