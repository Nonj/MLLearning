# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#print(dataset)

# x will always be matrix of independent variables (i.e. years of expr)
X = dataset.iloc[:, :-1].values

# y i is always the dependent variable (i.e. salary)
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#print(X_train)
#print(Y_train)

# Feature Scaling, Dont need to feature scale for this simple linear regression 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fits our regressor object to our training set 
# first parameter is matrix of features of training set (independent variables)
# second parameter is target values (dependent variables)
# Fitting means it learned the Correlation of our training set
# Meaning it learned how to predict salaries based on experience
regressor.fit(X_train, Y_train)

# Predicting the test set results
y_prediction = regressor.predict(X_test)
print(y_prediction)
print('\n')
print(Y_test)

# Visualizing the Training set of results
# X train is the X-coordinate of our observation (years of experience)
# Y train is the Y-coordinate of our observation (salary)
plt.scatter(X_train, Y_train, color = 'Red');

# X train is the X-coordinate of our observation (years of experience)
# regressor.predict(X_train) is the PREDICTED (salary) line in blue
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()