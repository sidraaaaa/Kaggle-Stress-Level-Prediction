#Q2-Modify the example keras code in class to work with the competition data.  
#Train the neural network both with and without dropout and check the accuracy on some held out data.  
#Which is more accurate?  Write a summary of your results and include your code in canvas.

#library(keras)
#library(tidyverse)
# install.packages("devtools")
# devtools::install_github("rstudio/keras")
# library(keras)

training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")
temp1<-training
##########Feature Scaling columns
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


X_join[,1]=(X_join[,1]-mean(X_join[,1]))/sd(X_join[,1]) #replacing NAs
#dim(X_join)

#Working with keras model with dropout rate

NN_keras_model<-keras_model_sequential()
NN_keras_model%>%
  layer_dense(units=50,activation='relu',
              input_shape = c(81))%>%
              layer_dropout(rate=0.5)%>%
              layer_dense(units=50,activation='relu')%>%
              layer_dropout(rate=0.5)%>%
              layer_dense(units=1,activation='linear')
  
summary(NN_keras_model)
NN_keras_model%>%compile(
        loss='MeanSquaredError',
        optimizer=optimizer_adam(),
        metrics=c('MeanSquaredError')
)  
history_X_join<-NN_keras_model%>%fit(X_join,y,
                     epochs=30, patch_size=128,
                     validation_split=0.2
                     ) #rmse= 2.828

plot(history_X_join)
model%>%predict(X_join)
  
#with dropout rate rmse=2.828


NN_keras_model_wo_dropout<-keras_model_sequential()
NN_keras_model_wo_dropout%>%
  layer_dense(units=50,activation='relu',
              input_shape = c(81))%>%
  layer_dense(units=50,activation='relu')%>%
  layer_dense(units=1,activation='linear')

summary(NN_keras_model_wo_dropout)

NN_keras_model_wo_dropout%>%compile(
  loss='MeanSquaredError',
  optimizer=optimizer_adam(),
  metrics=c('MeanSquaredError')
)

history_X_join<-NN_keras_model_wo_dropout%>%fit(X_join,y,
                                     epochs=30, patch_size=128,
                                     validation_split=0.2
) #rmse= 2.94

plot(history_X_join)
model%>%predict(X_join)

#with dropout rate in mode I was getting rmse= 2.828
#without dropout rate in mode I was getting rmse= 2.94
#It seems like including dropout rate is good because it
#doesnt allow model to overfit the training data