#Data Preprocessing

#Importing the Libraries
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

#Import datasetr
dataset = pd.read_csv('Data.csv');

# Taking all the columns of the dataset into x
# [:, :-1 ] // left colon means all of the lines, right 
# :-1 means takes all the columns except for last one
# So country, Age, Salay, and not purchased
x = dataset.iloc[:, :-1].values
#print(x)

# Creating dependent variable vector
y = dataset.iloc[:, 3].values
#print(y)

# Take care of missing data
from sklearn.preprocessing import Imputer

# will use imputer class to replace 'NaN' values with the mean
# of the columns (axis 0).
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# Call imputer's fit method on the dataset X, the first colon
# means all the lines, and the 1:3 means fix column 1 & 2 excluding 3
# since those contains the missing values
imputer = imputer.fit(x[:, 1:3])

# replace missing values with means of column
# we are taking columns where there is missing data in X
# The transform method actaully does the calculations on the columns for x
 
x[:, 1:3] = imputer.transform(x[:, 1:3])
#print(x)

# Have to encode categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_x = LabelEncoder();

# Encode the first column of x
# Problem is, it labels 0 , 1, 2  for spain france germany
# When we don't want any order that makes spain
# france or germany higher
x[:, 0] = label_encoder_x.fit_transform(x[:,0])
#print(x)

# This will create 3 columns for spain france and germany
# and will put a 1 or 0 on each line denoting if it is of a
# certain country
one_hot_encoder = OneHotEncoder(categorical_features = [0])
x = one_hot_encoder.fit_transform(x).toarray();
#print(x)

# This is binary so it is okay to just encode
label_encoder_purchased = LabelEncoder();
y = label_encoder_purchased.fit_transform(y)
#print(y)

#Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split

# Splitting up training/test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# checking number of train/test set 
# print(y_train)
# print(y_test)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

# Should fit on training set first so its scaled properly
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
