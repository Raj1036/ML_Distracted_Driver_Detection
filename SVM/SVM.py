
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.utils import shuffle
from skimage.feature import hog
from skimage import data, exposure
from skimage.transform import resize
import matplotlib.pyplot as plt

from sklearn.svm import SVC

import time
import cv2
import glob


def matrix(row, col, imgs):
    a = row * col *3
    vector_newX = np.zeros((a, 1))
    imgSeq = []
    for img in imgs:
        oriimg = cv2.imread(img)
        imgSeq.append(img.split('img_')[1].split('.jpg')[0])
        img0 = cv2.resize(oriimg, (row, col))
        flat = img0.reshape(a, 1)
        vector_newX = np.c_[vector_newX, flat]
        print(img)
    print(imgSeq[0], imgSeq[30], imgSeq[263])
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:, :]
    print('size of feature martix is:', np.shape(finalX_train))
    return finalX_train, imgSeq


def Hogmatrix(row, col, imgs):
    vector_newX = np.zeros((3888, 1))
    imgSeq = []
    for img in imgs:
        imgSeq.append(img.split('img_')[1].split('.jpg')[0])
        image = cv2.imread(img)
        resized_img = resize(image, (row, col))

        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)

        vector_newX = np.c_[vector_newX, fd]
        print(img)
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:, :]
    print('size of feature matrix is:', np.shape(finalX_train))
    return finalX_train, imgSeq


def svm():
    mydata = pd.read_csv(r"C:\Users\priya\Documents\Course material\Machine Learning\MLProject\venv\Include\trainHistograms.csv", header=None)

    X = mydata.iloc[0:, :-1].values
    Y = mydata.iloc[0:, -1].values
    print(np.shape(X))
    print(np.shape(Y))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=50)

    #svclassifier = SVC(kernel='rbf', gamma='scale', C=1)
    svclassifier = SVC(kernel='linear', C=1)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Validation Accuracy:", accuracy)

    plot_roc(y_test,y_pred)

    return svclassifier, accuracy


def plot_roc(y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area ={0:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def predict(X, imgSeq):
    Y_pred_case = svclassifier.predict(X)
    print('Predicted Result:', Y_pred_case)
    imgLabel = pd.read_csv(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\ImageLabels.csv")
    imgLabel["SVM"] = ''
    print(imgLabel.head(10))
    correct = 0
    for i, val in enumerate(imgSeq):
        imgLabel.loc[imgLabel["image"] == int(val), "SVM"] = Y_pred_case[i]
        if Y_pred_case[i] in str(imgLabel.loc[imgLabel["image"] == int(val)]["label"]):
            correct = correct + 1
    print(imgLabel.head(10))
    total = len(imgSeq)
    genAcc = (correct / total) * 100
    print("General Accuracy: ", genAcc)


# row = 64  # height of the image
# col = 64  # width of the image

row = 144  # height of the image
col = 192  # width of the image

imgs = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\Final_Test\*.jpg")
combined_train, imgSeq = Hogmatrix(row, col, imgs)
#combined_train, imgSeq = matrix(row, col, imgs)
svclassifier, acc = svm()
predict(combined_train, imgSeq)
start_time = time.time()
print("Accuracy:", acc)
print("--- %s seconds ---" % (time.time() - start_time))


