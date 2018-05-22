# Multiple Linear Regressio

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
#print(dataset)

# Getting independent Variables
X = dataset.iloc[:, :-1].values
#print(X)

# Grabbing dependent variabl column
y = dataset.iloc[:, 4].values

# Have to encode categorical variables in independant variabls
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()
#print(X)


#Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fit linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elmination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

# optimal matrix of features that will contain only 
# independant variables that matter to dependent variable
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

#fit full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
#print(regressor_OLS.summary())

#fit model without 2 because it has a significance level greater than 5%
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()

#print(regressor_OLS.summary())

#fit model without 1, 2 because it has a significance level greater than 5%
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
#print(regressor_OLS.summary())

#fit model without 1, 2, 4 because it has a significance level greater than 5%
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(regressor_OLS.summary())
print(X_opt)