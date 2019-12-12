import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
import time
import cv2
import glob

def matrix(row,col,imgs):
    a = row*col*3
    vector_newX = np.zeros((a, 1))
    imgSeq = []
    for img in imgs :
        oriimg = cv2.imread(img)
        imgSeq.append(img.split('img_')[1].split('.jpg')[0])
        img0 = cv2.resize(oriimg,(row,col))
        flat = img0.reshape(a,1)
        vector_newX = np.c_[vector_newX,flat]
        print(img)
    print(imgSeq[0],imgSeq[30],imgSeq[263])
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:,:]
    print('size of feature martix is:',np.shape(finalX_train))
    return  finalX_train,imgSeq

def tree():
    
    mydata = pd.read_csv(r"/home/nanikante1036/2NN/original_TrainData.csv",header = None)
    
    X = mydata.iloc[0:,:-1].values  #iloc is a --> Purely integer-location based indexing for selection by position from data.
    Y = mydata.iloc[0:,-1].values
    print(np.shape(X))
    print(np.shape(Y))
        
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 50)

    #Create Random Forest classifer object
    clf = RandomForestClassifier(n_estimators=50,criterion="entropy")
    
    # Train Random Forest Classifer
    clf = clf.fit(X_train,Y_train)
    
    #Predict the response for test dataset
    Y_pred = clf.predict(X_test)
    
    Accuracy = metrics.accuracy_score(Y_test, Y_pred)
    print("Split Accuracy:",(Accuracy*100))
    return clf,Accuracy

def predict(X,imgSeq):
    Y_pred_case = clf.predict(X)
    print('Predicted Result:',Y_pred_case)
    imgLabel = pd.read_csv("/home/nanikante1036/2NN/ImageLabels.csv")
    imgLabel["RandomForest_RGB"]=''
    print(imgLabel.head(10))
    correct=0
    for i,val in enumerate(imgSeq):
        imgLabel.loc[imgLabel["image"]==int(val),"RandomForest_RGB"]=Y_pred_case[i]
        if (Y_pred_case[i] in str(imgLabel.loc[imgLabel["image"]==int(val)]["label"])):
            correct = correct+1
    print(imgLabel.head(10))
    total = len(imgSeq)
    genAcc = (correct/total) *100
    print(correct, "Images classified correctly out of ",total, "images")
    print("General Accuracy: ",genAcc)
    imgLabel.to_csv("/home/nanikante1036/2NN/ImageLabels.csv")


row = 64        #height of the image 
col = 64        #width of the image    

imgs = glob.glob(r"/home/nanikante1036/2NN/test 500/*.jpg")
combined_train,imgSeq = matrix(row,col,imgs)
clf,acc = tree()
predict(combined_train,imgSeq)
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))