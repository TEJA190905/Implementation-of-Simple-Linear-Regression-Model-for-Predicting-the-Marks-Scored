# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## DATASET
![Screenshot (71)](https://github.com/user-attachments/assets/66cb55b4-9048-4f22-a5f7-5e4bbf7069d9)

## HEAD VALUES
![Screenshot (76)](https://github.com/user-attachments/assets/9af85c60-5f5d-44ea-9df9-2c2700540ac3)

## TAIL VALUES
![Screenshot (72)](https://github.com/user-attachments/assets/9235711a-cde1-4ef7-8dbf-31e48bfda55b)

## X AND Y VALUES
![Screenshot (73)(1)](https://github.com/user-attachments/assets/d0920126-89c6-415c-8fa2-f9f6953a64b2)

## PREDICTION VALUES OF X AND Y
![Screenshot (73)](https://github.com/user-attachments/assets/b7a9d0ee-e7e7-4838-a071-9cc868193226)

## TRAINING SET
![Screenshot (75)](https://github.com/user-attachments/assets/d7e3f30d-9071-4ec2-83e9-7e6c13c58af1)

## TESTING SET AND MSE,MAE AND MSE
![Screenshot (74)](https://github.com/user-attachments/assets/f6b098f4-6a79-4450-8c57-efc6cd3d8b84)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
