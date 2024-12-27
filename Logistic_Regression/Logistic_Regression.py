#!/home/pepito/Documents/ML/.venv/bin/python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def sigmoid(z):
    if z>0 :
        return 1/(1+np.exp(-z))
    else:
        return 1/(1+np.exp(z))

def P_Scalaire(X,theta):
    return np.dot(X,theta)

def Cross_Entropy(y_true,X,theta,biais):
    m= len(y_true)
    y_predit= P_Scalaire(X,theta) + biais
    cost=(-1/m)*np.sum(y_true*np.log(y_predit)+(1-y_true*np.log(1-h)))
    return cost

def Gradient_Descent(X,y_true,theta,L,iterations):
    m=len(y_true)
    for i in range(iterations):
        h=sigmoid(P_Scalaire(X,theta)+b)
        gradient= (1/m)*P_Scalaire(X.T,(h-y))
        theta -= gradient
        if iterations % 100 ==0:
            cost=Cross_Entropy(y_true,X,theta)
            print(f"It's the {i} th iteration and the cost is {cost} ")
    return theta
