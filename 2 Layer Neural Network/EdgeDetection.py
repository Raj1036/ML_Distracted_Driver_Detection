#import sys
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np			
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

row = 120        #height of the image																																																						
col = 120        #width of the image

def matrix(row,col,Y,images):
    a = row*col
    vector_newX = np.zeros((a, 1))
    vector_newY = []
    for img in images :
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        edges_high_thresh = cv2.Canny(gray, row, col)
        img0 = cv2.resize(edges_high_thresh,(row,col))
        flat = img0.reshape(a,1)
        vector_newX = np.c_[vector_newX,flat]
        vector_newY = np.append(vector_newY,Y)
        print(img)
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:,:]
    combined_train = np.c_[finalX_train,vector_newY]
    return  combined_train

# Y = [0]
# imgs = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c0/*.jpg")
# combined_train = matrix(row,col,Y,imgs)
#
# Y = [1]
# imgs1 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c1/*.jpg")
# combined_train1 = matrix(row,col,Y,imgs1)
# X1 = np.concatenate((combined_train,combined_train1))
#
# Y = [2]
# imgs2 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c2/*.jpg")
# combined_train2 = matrix(row,col,Y,imgs2)
# X2 = np.concatenate((X1,combined_train2))
#
# Y = [3]
# imgs3 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c3/*.jpg")
# combined_train3 = matrix(row,col,Y,imgs3)
# X3 = np.concatenate((X2,combined_train3))
#
# Y = [4]
# imgs4 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c4/*.jpg")
# combined_train4 = matrix(row,col,Y,imgs4)
# X4 = np.concatenate((X3,combined_train4))
#
# Y = [5]
# imgs5 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c5/*.jpg")
# combined_train5 = matrix(row,col,Y,imgs5)
# X5 = np.concatenate((X4,combined_train5))
#
# Y = [6]
# imgs6 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c6/*.jpg")
# combined_train6 = matrix(row,col,Y,imgs6)
# X6 = np.concatenate((X5,combined_train6))
#
# Y = [7]
# imgs7 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c7/*.jpg")
# combined_train7 = matrix(row,col,Y,imgs7)
# X7 = np.concatenate((X6,combined_train7))
#
# Y = [8]
# imgs8 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c8/*.jpg")
# combined_train8 = matrix(row,col,Y,imgs8)
# X8 = np.concatenate((X7,combined_train8))
#
# Y = [9]
# imgs9 = glob.glob(r"/media/arpit/New/Kratika/Sem1Courses/CS539_MachineLearning/project/state-farm-distracted-driver-detection/imgs/train/c9/*.jpg")
# combined_train9 = matrix(row,col,Y,imgs9)
# final_matrix = np.concatenate((X8,combined_train9))
#
# np.savetxt('final_train_gray2.csv',final_matrix, delimiter=',',fmt='%s')
#
# print('size of feature martix is:',np.shape(final_matrix))
import csv
with open("/media/raj/Rajendra/Rajendra/Courses/MS WPI/Fall'19/ML/Project_Distracted Driver Detection/Code/final_train_gray.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = list(reader)
    dataX0_X5 = np.array(data)
with open("/media/raj/Rajendra/Rajendra/Courses/MS WPI/Fall'19/ML/Project_Distracted Driver Detection/Code/final_train_gray2.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = list(reader)
    dataX6_X9 = np.array(data)
train_data0_9 = np.concatenate((dataX0_X5,dataX6_X9))
np.savetxt('edge_train.csv',train_data0_9, delimiter=',',fmt='%s')
