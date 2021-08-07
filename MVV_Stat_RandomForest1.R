#####################################################
###                                               ###
###      Multivariate Statistics                  ###
###  Weather Rain Forecasting and Modeling        ###
### Comparison Between ANN-MLP and Random Forest  ### 
###                                               ###
#####################################################
rm(list=ls())

# ______________________Required Packages & Libraries___________________________

# install.packages("neuralnet")
# install.packages("randomForest")
# install.packages("ROSE")


library("tidyverse")
library("caTools")
library("randomForest")
library("neuralnet")
library("ROSE")



# _______________________(1) DATA PREPROCESSING ________________________________

# First upload the data file and named it data-set
dataset <- read.csv(file = "weather_madrid_LEMD_1997_2015.csv", 
               header = TRUE,
               sep = ",")

# Descriptive Statistics of the given data-set.  
summary(dataset)
# Found some columns of the data-set have missing values (Not available data). 
                   

# Structure of the given data-set.
str(dataset)                       

# After review the structure of th data-set we change the Events columns from 
# character as factor to assigning the rain with binary outcomes of 0 and 1.

dataset$Events <- as.factor(dataset$Events)
levels(dataset$Events)
dataset$EventRain <- ifelse(dataset$Events == "Rain", 1,0)


# ______________________________(2) DATA PREPARATION ___________________________


# In new data frame (dataset1) we are selecting the 9 variables. 
dataset1 <- dataset %>% select(Max.TemperatureC, Min.TemperatureC, 
                          Max.DewpointC, Min.DewpointC, Max.Humidity, 
                          Min.Humidity, Max.Sea.Level.PressurehPa, 
                          Min.Sea.Level.PressurehPa, EventRain)


# Now we review descriptive summary of new data frame.
summary(dataset1)


# ______________________________(3) DATA CLEANING ______________________________


# In the new data frame (dataset2) removing the missing values (N/A's data).
dataset2 <- na.omit(dataset1)      


# In the new data frame without N/A's 'dataset2' having 6810 observation,
# with 1 dependent variable "EventRain" and others are independent.To run the 
# Logistic Regression on the binary outcome of the "Event Rain".
colnames(dataset2)


# Logistic Regression where "Event Rain"is a binary factor and x1-x8 are continuous predictors.
fit <- glm(EventRain ~ .,
           data=dataset2,
           family=binomial(link = "logit"))

summary(fit) # display results

# After applying Logistic model we have found some variables are highly 
# significant with variable of interest "Event Rain", 
# We select those independent variables for prediction.

dataset3 <- subset(dataset2, select = c(Max.TemperatureC, Min.TemperatureC, 
                                        Max.DewpointC, Min.Humidity, 
                                        Min.Sea.Level.PressurehPa, EventRain))


#Data set split to test and train data in the ratio 80:20%
# set seed
set.seed(123)
split = sample.split(dataset3$EventRain, SplitRatio = 0.8)
training_set = subset(dataset3, split == TRUE)
testing_set = subset(dataset3, split == FALSE)
training_set$EventRain

# Feature Scaling
training_set[-6] = scale(training_set[-6])
testing_set[-6] = scale(testing_set[-6])


# _____________________(4.1) MODELING RANDOM FOREST ____________________________


# set seed
set.seed(100)
# Apply the Random Forest Method
model_rf <- randomForest(EventRain ~ ., 
                         data = training_set, 
                         ntree = 200, 
                         mtry = 2,
                         importance = TRUE)

# view the model result
model_rf

# View the plot of the Random Forest Method.
plot(model_rf)

# Prediction of Random Forest model
pred1 <- predict(model_rf, 
                 newdata = testing_set[-6],
                 type = "class")

pred1
# Converting prediction into binary classes setting threshold level 0.5
y_pred_rf <- ifelse(pred1 > 0.5, 1, 0)
y_pred_rf


# Confusion Metrics of the random forest model
cm_rf <- table(testing_set[, 6], y_pred_rf)
cm_rf



# _______________(4.2) MODLING MLP- ARTIFICIAL NEURAL NETWORK __________________


#set seed
set.seed(100)
# Apply the MLP - Artificial Neural Network model
model_nn <- neuralnet(EventRain ~ .,
                      data = training_set, 
                      hidden = 4,
                      linear.output = FALSE,
                      stepmax = 1e6)


# plot the Artificial Neural Network
plot(model_nn)

# Prediction using Artificial Neural Network
predict <- compute(model_nn,testing_set)
predict$net.result


# Converting probabilities into binary classes setting threshold level 0.5
prob <- predict$net.result
pred2 <- ifelse(prob > 0.5, 1, 0)
pred2

# Confusion Metrics
cm_nn <- table(testing_set[, 6], pred2)
cm_nn


# _________________(5.1) Evaluation of Random Forest Model _____________________


# Accuracy 
accu_model_rf <- ((cm_rf[1,1] + cm_rf[2,2])/
  (cm_rf[1,1] + cm_rf[2,2] + cm_rf[2,1] + cm_rf[1,2]))*100
accu_model_rf

# Precision 
precision_rf <- (cm_rf[1,1]/(cm_rf[1,1]+cm_rf[1,2]))*100
precision_rf

# Recall 
recall_rf <- (cm_rf[1,1]/(cm_rf[1,1]+cm_rf[2,1]))*100 
recall_rf

# F-Score 
f_score_rf<- (cm_rf[1,1]/(cm_rf[1,1] + (0.5*(cm_rf[1,2]+cm_rf[2,1]))))*100
f_score_rf

# ROC curve
par(mfrow = c(2, 2))
roc_rf <- roc.curve(testing_set$EventRain, y_pred_rf, 
                    main ="ROC curve \n (Random Forest)")
                
roc_rf



# ___________(5.2) Evaluation of Artificial Neural Network Model _______________


# Accuracy
accu_model_nn <- ((cm_nn[1,1] + cm_nn[2,2])/
  (cm_nn[1,1] + cm_nn[2,2] + cm_nn[2,1] + cm_nn[1,2]))*100
accu_model_nn

# Precision 
precision_nn <- (cm_nn[1,1]/(cm_nn[1,1]+cm_nn[1,2]))*100
precision_nn

# Recall 
recall_nn <- (cm_nn[1,1]/(cm_nn[1,1]+cm_nn[2,1]))*100 
recall_nn

# F-Score
f_score_nn <- (cm_nn[1,1]/(cm_nn[1,1] + (0.5*(cm_nn[1,2]+cm_nn[2,1]))))*100
f_score_nn

#ROC curve 
roc_nn <- roc.curve(testing_set$EventRain, pred2,
                    main ="ROC curve \n (ANN-MLP)")
roc_nn


# view random forest plot
plot(model_rf)



# _______________(6) Performance of the Both Models Comparison _________________

# Comparison of the Random Forest and Artificial Neural Network-MLP
Results <- data.frame(Models = c("Random Forest", "ANN-MLP"),
                      Accuray = c(accu_model_rf, accu_model_nn), 
                      Precision = c(precision_rf, precision_nn),
                      Recall = c(recall_rf, recall_nn),
                      F_Score = c(f_score_rf, f_score_nn),
                      ROC_AUC = c((roc_rf$auc)*100, (roc_nn$auc)*100))

Results


# ___________________(7) Summary of the Results of Model _______________________

# Artificial Neural Network performance better in terms of model prediction accuracy
# 88% is accurate than Random Forest. Also performs well in other measures
# such as Precision, Recall, F-Score, ROC (AUC)



