import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
from datetime import datetime
datadirectory="F:\\Folder_3\\downloads!\\cat-and-dog\\training_set\\training_set"
testdirectory="F:\\Folder_3\\downloads!\\cat-and-dog\\test_set\\test_set"
categories=["cats","dogs"]
training_data=[]
testing_data=[]
for category in categories:
    path=os.path.join(datadirectory,category)
    class_num=categories.index(category)
    
    for img in os.listdir(path):
        
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array=cv2.resize(img_array,(50,50))
            new_array=new_array.flatten()
            new_array=new_array.reshape(2500,1)
            #print(new_array)
            #print(new_array.shape)
           
            training_data.append([new_array,class_num])
           
            #plt.imshow(new_array,cmap="gray")
            #plt.show()
           
        except Exception as e:
            pass
for category in categories:
    path=os.path.join(testdirectory,category)
    class_num=categories.index(category)
    
    for img in os.listdir(path):
        
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array=cv2.resize(img_array,(50,50))
            new_array=new_array.flatten()
            new_array=new_array.reshape(2500,1)
            #print(new_array)
            #print(new_array.shape)
           
            testing_data.append(new_array)
           
            #plt.imshow(new_array,cmap="gray")
            #plt.show()
           
        except Exception as e:
            pass
lr=0.001
training_data=np.array(training_data)
testing_data=np.array(testing_data)
#testing_data=testing_data.reshape(2023,2500)
testing_data=testing_data.T


#print(training_data)
#print(type(img_array))

random.shuffle(training_data)
#random.shuffle(testing_data)



X=[]
Y=[]
for features,label in training_data:
    #print(features.shape)
    X.append(features)
    Y.append(label)
  
X=np.array(X).reshape(-1,2500)
Y=np.array(Y).reshape(-1,1)
#print(X)
X=X.T

W1=np.random.randn(2500,2500)*.001
B1=np.zeros(1)
W2=np.random.randn(2500,1)*.001
B2=np.zeros(1)


Y=Y.T

print(datetime.now())

for i in range(10):
#Feedforward Propogation
    Z1=np.dot(W1.T,X)+B1
    A1=Z1
    Z2=np.dot(W2.T,A1)+B2
    A2=1/(1+np.exp(-1*Z2))
#Backward propagation

    dZ2=A2-Y
    dW2=(np.dot(dZ2,A1.T))/94
    dB2=(np.sum(dZ2,axis=1,keepdims=True))/94
    dZ1=A1-Y
    dW1=np.dot(dZ1,X.T)/94
    dB1=(np.sum(dZ1,axis=1,keepdims=True))/94
   
    W1=W1-dW1.T*lr
    B1=B1-dB1*lr
    W2=W2-dW2.T*lr
    B2=B2-dB2*lr
   
    L=-1*(np.dot(Y,np.log(A2.T))+np.dot((1-Y),np.log(1-(A2.T))))
    print(L)
print(L)

Z1=np.dot(W1.T,testing_data)+B1
A1=Z1
Z2=np.dot(W2.T,A1)+B2
A2=1/(1+np.exp(-1*Z2))
print(A2)

print(datetime.now())
