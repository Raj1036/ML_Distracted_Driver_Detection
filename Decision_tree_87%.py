import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2
import glob

##############GENERALIZE CHECKING DATA GENERATE#################################

row = 64
col = 64
a = row*col*3

oimg1 = cv2.imread(r"D:\Kaggle_Distracted_Driver\Check_vectorize\img_104.jpg")
img1 = cv2.resize(oimg1,(row,col))

vector_newX = np.reshape(img1,(row*col*3,1))

imgs = glob.glob(r"D:\Kaggle_Distracted_Driver\Generalize_images\*.jpg")

for img in imgs:
    oriimg = cv2.imread(img)
    img0 = cv2.resize(oriimg,(row,col))
    flat = img0.reshape(a,1)
    vector_newX = np.c_[vector_newX,flat]
    print(img)

vector_newX = vector_newX.T

finalX_gen = vector_newX[1:,:]

###################IMPORT TRAINING DATA##########################################
data = pd.read_csv(r'D:\Kaggle_Distracted_Driver\Pracrice images\full_64x64.csv'  , header = None)
print(data)

Xo = data.drop(data.columns[-1], axis=1)
print(Xo)

Yo = data[data.columns[-1]]
print(Yo)

X_train,X_test,y_train,y_test=train_test_split(Xo , Yo , test_size = 0.15, random_state = 100)
#print(X_train)
#print(y_train)
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

y_generalize = classifier.predict(finalX_gen)

print(y_generalize)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

################################################################################



'''oimg1 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_7.jpg")
img1 = cv2.resize(oimg1,(64,64))

vector_newX1 = np.reshape(img1,(64*64*3,1))
vector_newX1 = vector_newX1.T

y_new1 = classifier.predict(vector_newX1)
print("The actual class is c0, predicted class for first real world image is: ",y_new1)

oimg2 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_8.jpg")
img2 = cv2.resize(oimg2,(64,64))

vector_newX2 = np.reshape(img2,(64*64*3,1))
vector_newX2 = vector_newX2.T

y_new2 = classifier.predict(vector_newX2)
print("The actual class is c3, predicted class for second world image is: ",y_new2)

oimg3 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_9.jpg")
img3 = cv2.resize(oimg3,(64,64))

vector_newX3 = np.reshape(img3,(64*64*3,1))
vector_newX3 = vector_newX3.T

y_new3 = classifier.predict(vector_newX3)
print("The actual class is c5, predicted class for third world image is: ",y_new3)

oimg4 = cv2.imread(r"D:\Kaggle_Distracted_Driver\imgs\test\img_11.jpg")
img4 = cv2.resize(oimg4,(64,64))

vector_newX4 = np.reshape(img4,(64*64*3,1))
vector_newX4 = vector_newX4.T

y_new4 = classifier.predict(vector_newX4)
print("The actual class is c1, predicted class for fourth real world image is: ",y_new4)'''

