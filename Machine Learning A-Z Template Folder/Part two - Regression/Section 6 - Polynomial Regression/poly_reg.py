# Ploynomial Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
#print(X_poly)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing linear Regression Results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel("Position Level")
plt.ylabel('Salary')
#plt.show()
 

# Visualizing Polynomial Regression Results
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = "blue")
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel("Position Level")
plt.ylabel('Salary')
#plt.show()

# Predicting new Result with Linear Regression
#print(lin_reg.predict(6.5))

# Predicitng a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))