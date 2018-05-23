#Multiple Linear Regr

#Importing the dataset
dataset = read.csv('50_Startups.csv')


# Encoding categorical data
dataset$State = factor(dataset$State, 
                         levels = c('New York', 'California', 'Florida'), 
                         labels = c(1, 2, 3))

# Split the dataset into the Training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

# Fitting multiple Linear Regression to the Training Set
# Profit ~ R.D.Spend + Administration + Marketing.Spend + State
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the test set results
y_pred = predict(regressor, newdata = test_set)


# building the optimal model using Backward Elinmination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)

summary(regressor)