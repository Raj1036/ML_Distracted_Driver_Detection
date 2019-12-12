import cv2
import numpy as np
import glob

from sklearn.preprocessing import MinMaxScaler


def matrix(row, col, Y, imgs):
    a = row * col * 3
    vector_newX = np.zeros((a, 1))
    vector_newY = []
    for img in imgs:
        oriimg = cv2.imread(img)
        img0 = cv2.resize(oriimg, (row, col))
        flat = img0.reshape(a, 1)
        vector_newX = np.c_[vector_newX, flat]
        vector_newY = np.append(vector_newY, Y)
        print("reading...")
    vector_newX = vector_newX.T
    finalX_train = vector_newX[1:, :]

    # scale = MinMaxScaler(feature_range=(0, 1))
    # finalX_train = scale.fit_transform(finalX_train)

    combined_train = np.c_[finalX_train, vector_newY]
    # print('size of feature martix is:',np.shape(combined_train))
    return combined_train


row = 64  # height of the image
col = 64  # width of the image

Y = ['c0']
imgs = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c0\*.jpg")
combined_train = matrix(row, col, Y, imgs)
np.savetxt('train.csv', combined_train, delimiter=',', fmt='%s')

Y = ['c1']
imgs1 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c1\*.jpg")
combined_train1 = matrix(row, col, Y, imgs1)
# X1 = np.concatenate((combined_train,combined_train1))
with open('train.csv', 'ab') as f:
    np.savetxt(f, combined_train1, delimiter=',', fmt='%s')

Y = ['c2']
imgs2 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c2\*.jpg")
combined_train2 = matrix(row, col, Y, imgs2)
# X2 = np.concatenate((X1,combined_train2))

with open('train.csv', 'ab') as f:
    np.savetxt(f, combined_train2, delimiter=',', fmt='%s')

Y = ['c3']
imgs3 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c3\*.jpg")
combined_train3 = matrix(row, col, Y, imgs3)
# X3 = np.concatenate((X2,combined_train3))

with open('train.csv', 'ab') as f:
    np.savetxt(f, combined_train3, delimiter=',', fmt='%s')

Y = ['c4']
imgs4 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c4\*.jpg")
combined_train4 = matrix(row, col, Y, imgs4)
# X4 = np.concatenate((X3,combined_train4))

with open('train.csv', 'ab') as f:
    np.savetxt(f, combined_train4, delimiter=',', fmt='%s')

Y = ['c5']
imgs5 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c5\*.jpg")
combined_train5 = matrix(row, col, Y, imgs5)
# X5 = np.concatenate((X4,combined_train5))

with open('train.csv', 'ab') as f:
    np.savetxt(f, combined_train5, delimiter=',', fmt='%s')


Y = ['c6']
imgs6 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c6\*.jpg")
combined_train6 = matrix(row,col,Y,imgs6)
# X6 = np.concatenate((X5,combined_train6))

with open('train.csv','ab') as f:
    np.savetxt(f,combined_train6, delimiter=',',fmt='%s')

Y = ['c7']
imgs7 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c7\*.jpg")
combined_train7 = matrix(row,col,Y,imgs7)
# X7 = np.concatenate((X6,combined_train7))

with open('train.csv','ab') as f:
   np.savetxt(f,combined_train7, delimiter=',',fmt='%s')

Y = ['c8']
imgs8 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c8\*.jpg")
combined_train8 = matrix(row,col,Y,imgs8)
# X8 = np.concatenate((X7,combined_train8))

with open('train.csv','ab') as f:
   np.savetxt(f,combined_train8, delimiter=',',fmt='%s')

Y = ['c9']
imgs9 = glob.glob(r"C:\Users\priya\Documents\Course material\Machine Learning\Project\Dataset\imgs\train\c9\*.jpg")
combined_train9 = matrix(row,col,Y,imgs9)

with open('train.csv','ab') as f:
   np.savetxt(f,combined_train9, delimiter=',',fmt='%s')

print('Done!')


