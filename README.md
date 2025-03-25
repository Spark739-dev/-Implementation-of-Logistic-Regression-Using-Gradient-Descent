# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python libraries to perform the given question.
2. Use the placement data and perform the gradient descent for logistic regression for the given question.
3. Use train test split to categorize the data for x and y values
4. Use the fit for x_train and y_train.
5. print the y_pred in order to get predicted value using predict for x_test.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VESHWANTH.
RegisterNumber: 212224230300
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\admin\\OneDrive\\Documents\\My Games\\Placement_Data.csv")
data
data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)
data["ssc_b"]=data["ssc_b"].astype('category')
data["gender"]=data["gender"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')

data["ssc_b"]=data["ssc_b"].cat.codes
data["gender"]=data["gender"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes

theta = np.random.randn(x.shape[1]) 
Y=y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,Y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta  
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy = ",accuracy)
print(y_pred)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print(y_prednew)
  
*/
```

## Output:
![1](https://github.com/user-attachments/assets/e95b61d5-1ee1-472a-9bad-06a502c7b98e)
![Screenshot 2025-03-25 090120](https://github.com/user-attachments/assets/b7de71c7-b81b-47d1-9ddf-1b7ef48d40c9)
![3](https://github.com/user-attachments/assets/8c948249-270a-4a95-a568-2c0818a7e0ff)
![4](https://github.com/user-attachments/assets/5395b2df-d9a6-4cdf-8db3-4d771b792b54)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

