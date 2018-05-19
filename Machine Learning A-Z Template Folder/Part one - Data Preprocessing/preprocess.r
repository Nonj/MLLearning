#data preprocess

#Importing the dataset
dataset = read.csv('Data.csv')

# Replacing missing value in Age column with AVG of entire column
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

# Replacing missing value in Salary column with AVG of entire column
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)


# Encoding categorical data
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'), 
                         labels = c(1, 2, 3))

dataset$Purchased= factor(dataset$Purchased, 
                         levels = c('No', 'Yes'), 
                         labels = c(0,1))


# Split the dataset into the Training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)


# Featured Scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])





