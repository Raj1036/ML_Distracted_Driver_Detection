import cv2 
import numpy as np 
import glob

def matrix(row,col,Y,imgs):
    a = row*col
    DDEPTH = cv2.CV_16S
    vector_newX = np.zeros((a, 1))
    vector_newY = []
    
    for img_path in imgs:
        oriimg = cv2.imread(img_path,0)
        img0 = cv2.resize(oriimg,(row,col))
        img = cv2.GaussianBlur(img0, (3, 3), 2)

        gradx = cv2.Sobel(img, DDEPTH , 1, 0, ksize=3, scale=1, delta=0)
        gradx = cv2.convertScaleAbs(gradx)
        
        grady = cv2.Sobel(img, DDEPTH , 0, 1, ksize=3, scale=1, delta=0)
        grady = cv2.convertScaleAbs(grady)
        
        grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
        print(np.shape(grad))
        flat = grad.reshape(a,1)
        vector_newX = np.c_[vector_newX,flat]
        vector_newY = np.append(vector_newY,Y)
        print(img_path)
    
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:,:]
    combined_train = np.c_[finalX_train,vector_newY]
    print('size of feature martix is:',np.shape(combined_train))
    return  combined_train

row = 100        #height of the image 
col = 100        #width of the image    

Y = ['c0']
imgs = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c0\*.jpg")
#print("h",imgs[0])
combined_train = matrix(row,col,Y,imgs)


Y = ['c1']
imgs1 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c1\*.jpg")
combined_train1 = matrix(row,col,Y,imgs1)
X1 = np.concatenate((combined_train,combined_train1))

Y = ['c2']
imgs2 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c2\*.jpg")
combined_train2 = matrix(row,col,Y,imgs2)
X2 = np.concatenate((X1,combined_train2))

Y = ['c3']
imgs3 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c3\*.jpg")
combined_train3 = matrix(row,col,Y,imgs3)
X3 = np.concatenate((X2,combined_train3))

Y = ['c4']
imgs4 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c4\*.jpg")
combined_train4 = matrix(row,col,Y,imgs4)
X4 = np.concatenate((X3,combined_train4))

Y = ['c5']
imgs5 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c5\*.jpg")
combined_train5 = matrix(row,col,Y,imgs5)
X5 = np.concatenate((X4,combined_train5))

Y = ['c6']
imgs6 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c6\*.jpg")
combined_train6 = matrix(row,col,Y,imgs6)
X6 = np.concatenate((X5,combined_train6))

Y = ['c7']
imgs7 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c7\*.jpg")
combined_train7 = matrix(row,col,Y,imgs7)
X7 = np.concatenate((X6,combined_train7))
#
Y = ['c8']
imgs8 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c8\*.jpg")
combined_train8 = matrix(row,col,Y,imgs8)
X8 = np.concatenate((X7,combined_train8))

Y = ['c9']
imgs9 = glob.glob(r"H:\Masters Study\Machine Learning\Project\Kaggle_Dataset\c9\*.jpg")
combined_train9 = matrix(row,col,Y,imgs9)
final_matrix = np.concatenate((X8,combined_train9))

np.savetxt('Sobel_train.csv',final_matrix, delimiter=',',fmt='%s')

print('size of feature martix is:',np.shape(final_matrix))


#print(combined_train[:,13440])

##################  Checking each features with images: #######################
#check1 = finalX_train[0]
#check = check1.reshape(col,row,3)

#oimg = cv2.imread(r"C:\Users\vrush\Jupyter Noteboks\ML\img_104.jpg")
#img1 = cv2.resize(oimg,(row,col))
#print(img1)
#print(np.array_equal(check, img1))
###############################################################################