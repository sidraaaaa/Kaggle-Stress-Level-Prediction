###################GLMNET MODEL
#setwd("HW 6/ Kaggle-Stress-Level-Prediction")

training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")
index <- sample(2,nrow(training),replace=TRUE,prob=c(0.5,0.5))
training[is.na(training)]=-1
testing[is.na(testing)]=-1

temp1<-training[index==1,]
temp2<-training[index==2,]

#Q1.  Fit a sparse regression to the competition data using glmnet with 
# alpha=1.  Include all columns that are either numeric or categorical 
# with a limited number of categories (so remove columns like ID, date that 
#                                      have many unique values).  Tune 
# lambda so that you end up with only 10 coefficients that are non-zero.  
# What are the 10 variables that those correspond to and are they positively 
# or negatively correlated with stress? 





