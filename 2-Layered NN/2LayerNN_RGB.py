import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

row=64
col=64
mydata = pd.read_csv(r"/home/nanikante1036/2NN/original_TrainData.csv",header = None)

Input = mydata.iloc[0:,:-1].values/255.0
Label = mydata.iloc[0:,-1].values
for i in range(len(Label)):
    Label[i] = int(Label[i][1])
    
X=Input.copy()
Y=Label.copy()

X = np.array(list(x for x in X))
Y = np.array(list(x for x in Y))

#print(Y[0:5])
Y=Y.T
#print(Y.shape)
#print(X.shape)
#print(X[0])
#print(X[0:3])
#type(X), type(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 50)

#class_names = [0,1,2,3,4,5,6,7,8,9]

#print(Y_train[0:5])
type(X_train), type(Y_train), type(X_test), type(Y_test)

model = keras.Sequential([keras.layers.Flatten(),
                          keras.layers.Dense(256, activation=tf.nn.relu),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=10)

test_loss,test_acc = model.evaluate(X_test,Y_test)
print('Split Test accuracy: ',test_acc)
predictions = model.predict(X_test)
print(predictions[0])
print(np.argmax(predictions[0]), ':',Y_test[0])

#from sklearn import metrics
import cv2
import glob

def matrix(row,col,imgs):
    a = row*col*3
    vector_newX = np.zeros((a, 1))
    imgSeq = []
    for img in imgs :
        image = cv2.imread(img)/255.
        imgSeq.append(img.split('img_')[1].split('.jpg')[0])
        img0 = cv2.resize(image,(row,col))
        flat = img0.reshape(a,1)
        vector_newX = np.c_[vector_newX,flat]
        print(img)
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:,:]
    print('size of feature martix is:',np.shape(finalX_train))
    return  finalX_train,imgSeq
testimgs = glob.glob(r"/home/nanikante1036/2NN/test 500/*.jpg")
combined_train,imgSeq = matrix(row,col,testimgs)

General_prediction = model.predict(combined_train)
Y_pred_case = np.empty(General_prediction.shape[0],dtype=object)
print(len(General_prediction))
for i in range(len(General_prediction)):
    l = np.argmax(General_prediction[i])
    Y_pred_case[i] = 'c'+ str(l)
print(Y_pred_case)
imgLabel = pd.read_csv("/home/nanikante1036/2NN/ImageLabels.csv")
imgLabel["NeuralNet_Gray"]=''
print(imgLabel.head(10))
correct=0
for i,val in enumerate(imgSeq):
    imgLabel.loc[imgLabel["image"]==int(val),"NeuralNet_Gray"]=Y_pred_case[i]
    if (Y_pred_case[i] in str(imgLabel.loc[imgLabel["image"]==int(val)]["label"])):
        correct = correct+1
print(imgLabel.head(10))
total = len(imgSeq)
genAcc = (correct/total) *100
print(correct, "Images classified correctly out of ",total, "images")
print("General Accuracy: ",genAcc)
imgLabel.to_csv("/home/nanikante1036/2NN/ImageLabels.csv",index=False)
#41/120 correct






