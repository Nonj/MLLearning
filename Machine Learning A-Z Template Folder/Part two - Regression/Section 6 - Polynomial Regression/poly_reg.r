#Data preprocess template

#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


# Split the dataset into the Training set and test set
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split==TRUE)
# test_set = subset(dataset, split==FALSE)


# Featured Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)


# Fitting Polynomial Regression to Dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level6 = dataset$Level^6

poly_reg = lm(formula = Salary ~ ., data = dataset)


#Visualizing Linear Regression Results
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color = 'red') +
  geom_line(aes(x=dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth Or Bluff') +
  xlab("Level") +
  ylab("Salary")


# Visualizing Poly Regression To Dataset
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), 
             color = 'red') +
  geom_line(aes(x=dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth Or Bluff (Poly)') +
  xlab("Level") +
  ylab("Salary")


# Predicting a new result with linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Poly Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5, 
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4,
                                      Level6 = 6.5^6))





