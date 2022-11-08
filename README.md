# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VARSHA AJITH
RegisterNumber:  212221230118
*/
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

## Output:
![op1](https://user-images.githubusercontent.com/94222288/200590486-253273eb-3d8f-4514-9e1f-94f2bc7a2142.png)
![op2](https://user-images.githubusercontent.com/94222288/200590551-8b44e4fc-ad10-4751-8c91-7cabdadf4102.png)
![op3](https://user-images.githubusercontent.com/94222288/200590574-c6aa268d-da12-445f-9574-1f3fe6dddf47.png)
![op4](https://user-images.githubusercontent.com/94222288/200590592-688e3d50-7fe6-412b-8c0c-7f909c069a2e.png)

![op5](https://user-images.githubusercontent.com/94222288/200590614-5b697a98-1c96-4411-a65f-4619e01472f4.png)
![op6](https://user-images.githubusercontent.com/94222288/200590685-4e19ebd1-12ab-4372-818b-d16e1b2d2e87.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

