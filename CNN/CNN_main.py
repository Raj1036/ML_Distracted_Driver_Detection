import cv2  
import numpy as np 
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    #X_Y_data()    ## turn on this function to create the data files 
    
    # if you run this code for the forst time run the above function and create npy array anf then run the other part
    
    X_data,Y_label,X_pred_general = Load('X.npy','Y_int.npy','X_pred1.npy')
    X_N,X_val,Y_N,Y_val = train_test_split(X_data,Y_label,test_size = 0.1,random_state = 50)
    
    X_train,X_test,Y_train,Y_test = train_test_split(X_N,Y_N,test_size = 0.2,random_state = 50)
    
    print('X_train shape:', np.array(X_train).shape)
    print('Y_train shape:', np.array(Y_train).shape)
    print('X_test shape:', np.array(X_test).shape)
    print('Y_test shape:', np.array(Y_test).shape)
    print('X_Val shape:', np.array(X_val).shape)
    print('Y_Val shape:', np.array(Y_val).shape)
    
    history = CNN_fit(X_train,X_test,Y_train,Y_test)
        
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#    Evaluate(model,X_test,Y_test)
#    Evaluate(model,X_val,Y_val)
#    Y_pred = Predict(model,X_val)
#    General_Acc(Y_pred,Y_val)
    
def X_Y_data():
    X_data = []
    Y_label = []
    Y_label_int = []
    i = 10
    row = 224        #height of the image 
    col = 224        #width of the image    
    
    for n in range(i):
        w = "H:\Masters Study\Machine Learning\Project\state-farm-distracted-driver-detection\imgs\\train\c"+str(n)+"\*.jpg"
        print(w)
        e = "c"+str(n)
        
        files = glob.glob (w)
        for myFile in files:
            image = cv2.imread (myFile) 
            img = cv2.resize(image,(row,col))
            X_data.append (img)
            Y_label.append(e)
            Y_label_int.append(n)
    X_data = np.array(X_data)
    Y_label = np.array(Y_label)
    Y_label_int = np.array(Y_label_int)
    print(Y_label_int)
    np.save('X_vgg', X_data)
    np.save('Y_vgg', Y_label)
    np.save('Y_int_vgg', Y_label_int)
    


def Load(x_name,y_name,xp_name):
    X_data = np.load(x_name)
    Y_label = np.load(y_name) 
    X_Pred = np.load(xp_name)
    #p = X_data[14440,:,:,:]
    #q = Y_label[14440,]
    #print(q)
    #cv2.imshow('img',p)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return X_data,Y_label,X_Pred

def CNN_fit(X_train,X_test,Y_train,Y_test):
    X_train= X_train/ 255.0
    X_test= X_test/ 255.0
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
       tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, epochs=5)
    history = model
    return history

def Evaluate(model,X_test,Y_test):    
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
        
def Predict(model,X_input):
    X_input = X_input/255.0
    Y_pred = model.predict(X_input, verbose=0)
    y = np.zeros(Y_pred.shape[0])
    for i in range(len(Y_pred)):
        y[i] = np.argmax(Y_pred[i])
    #print(y)
    return y

def General_Acc(Y_pred,Y_act):
    
#    Y_act = np.array([5,5,0,0,0,5,0,0,2,1,6,0,0,7,7,2,4,1,5,3,6,3,0,3,3,2,1,4,0,1,7,4,0,0,1,5,2,3,4,3,7,4,3,6,6,8,5,6,2,5,
#    6,0,0,0,2,3,1,6,4,7,0,4,2,1,9,1,4,1,6,9,0,6,8,9,5,5,6,0,2,0,0,1,9,6,
#    3,7,0,5,7,7,4,6,6,3,3,1,1,0,3,3,5,3,6,5,8,0,7,2,2,6,3,5,0,4,6,2,4,1,6,5])
    
    x_C = np.subtract(Y_pred,Y_act)
    #array_unique_ele,ele_count = np.unique(x_C,return_counts = True)
    incorrect = np.count_nonzero(x_C)
    correct = np.size(Y_act) - incorrect
    General_acc = (correct/np.size(Y_act))*100
    print("General Accuracy :", General_acc)
    return  General_acc



if __name__=="__main__":
    main()