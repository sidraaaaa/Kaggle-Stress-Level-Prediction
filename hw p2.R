###################GLMNET MODEL
#setwd("HW 6/ Kaggle-Stress-Level-Prediction")
#Q1.  Fit a sparse regression to the competition data using glmnet with 
# alpha=1.  Include all columns that are either numeric or categorical 
# with a limited number of categories (so remove columns like ID, date that 
#                                      have many unique values).  Tune 
# lambda so that you end up with only 10 coefficients that are non-zero.  
# What are the 10 variables that those correspond to and are they positively 
# or negatively correlated with stress? 

#Working with numerical data columns first
#str(temp1)
#library(glmnet)

training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")
index <- sample(2,nrow(training),replace=TRUE,prob=c(0.5,0.5))
temp1<-training[index==1,]
temp2<-training[index==2,]
# training[is.na(training)]=-1
# testing[is.na(testing)]=-1

y=temp1$pstr         #holding out y label
X=temp1[,3:84]      # for glmnet's numeric matrix
X$pstr=NULL         #removing pstr
X$SEX=as.numeric(as.factor(X$SEX))   #factoring the categorical value
pstr_table=temp1%>%group_by(higheduc)%>%summarise(medianpstr=median(pstr)) #substituting median of pstr for higheduc:categorical column
X_join=right_join(X,pstr_table,by="higheduc") #joining both tables
#head(X_join$medianpstr) 
table(X_join$medianpstr)
X_join$higheduc=NULL #higheduc=NA
X_join[is.na(X_join)]=0 #putting zero in place of all NAs
X_join=as.matrix(X_join) #making matrix
temp2_mat<-temp2[,3:84]               #repeating the same thing for cross-validated set
temp2_mat$SEX=as.numeric(as.factor(temp2_mat$SEX))
temp2_mat$higheduc=NULL
temp2_mat[is.na(temp2_mat)]=0
temp2_mat=as.matrix(temp2_mat)



model_glmsparse<-glmnet(X_join,y,alpha=1) #glmnet model with alpha 1
plot(model_glmsparse)                     #plotting 
# sum(coef(model_glmsparse,s=0.17)==0) 
# coef(model_glmsparse,s=0.17)


cv <- cv.glmnet(X_join,y)                 #finding best alpha for our glmnet model
modell=glmnet(X_join,y,lambda=cv$lambda.1se)  # cv$lambda.1se = 0.1209962 recommended way by authors to pick lambda
sum(coef(modell,s=cv$lambda.1se)==0)  #total 65 zero columns

pred_model<-predict(modell,newx=temp2_mat,s=cv$lambda.1se)

rmse(temp2$pstr,pred_model) #2.85











