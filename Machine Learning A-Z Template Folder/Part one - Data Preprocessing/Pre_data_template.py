# Template for data pre processing

#Importing the Libraries
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


#Import dataset
dataset = pd.read_csv('Data.csv');
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
# Splitting up training/test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
# Should fit on training set first so its scaled properly
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
