# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1 Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students

2 Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored

3 Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis

4 Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b for each data point calculate the difference between the actual and predicted marks

5 Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error

6 Once the model parameters are optimized, use the final equation to predict marks for any new input data
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harisudhan S
RegisterNumber:  212224240048


/*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
*/
```

## Output:

HEAD VALUES:

<img width="167" height="241" alt="image" src="https://github.com/user-attachments/assets/e6cb3c49-864f-48f3-81f0-1ec2ef0ee30e" />

TAIL VALUES:

<img width="178" height="235" alt="image" src="https://github.com/user-attachments/assets/450d8d35-0158-4295-b059-1e29878c93cc" />

X VALUES:

<img width="160" height="557" alt="image" src="https://github.com/user-attachments/assets/a5ecc66d-9400-496d-93c2-00350fa28363" />

Y VALUES:

<img width="718" height="60" alt="image" src="https://github.com/user-attachments/assets/a6873ba3-8de0-4d6e-99f5-2c30a045e60d" />

ACTUAL VALUES:

<img width="576" height="28" alt="image" src="https://github.com/user-attachments/assets/f7b93bb4-98d8-4946-966e-9e64775ec102" />

PREDICTED VALUES:

<img width="698" height="74" alt="image" src="https://github.com/user-attachments/assets/a77a7976-5693-4264-84af-454167e04303" />

TRAINING SET:

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/83f9c68c-41d2-497a-bd01-a208bd618488" />

TESTING SET:

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/6e08cf4c-413d-4355-9474-1e05f01e1fa6" />


MSE, MAE , RMSE:

<img width="258" height="66" alt="image" src="https://github.com/user-attachments/assets/f524b1f4-8226-4601-b6d1-e8209def29f9" />





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
