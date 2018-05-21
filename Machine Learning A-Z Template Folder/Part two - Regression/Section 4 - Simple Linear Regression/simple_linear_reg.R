# Simple Linear Regression
#Data preprocess template

#Importing the dataset
dataset = read.csv('Salary_Data.csv')


# Split the dataset into the Training set and test set
library(caTools)
set.seed(123)
dependent_v = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, dependent_v==TRUE)
test_set = subset(dataset, dependent_v==FALSE)


# Featured Scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Simple Linear Regression to the Training Set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)


# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)


# Visualizing the Training set results ***cmd + shift + c to comment***
# library(ggplot2)

# PLot training observation points
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') + 
  ggtitle('Salary vs. Experience (Training Set)') +
  xlab('Years of Experience') + 
  ylab('Salay')

# PLot TEST observation points
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') + 
  ggtitle('Salary vs. Experience (Test Set)') +
  xlab('Years of Experience') + 
  ylab('Salay')










