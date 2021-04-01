install.packages('e1071')
library(e1071)

training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")
index=sample(nrow(training),nrow(training)/2,F)
temp1<-training[index,] #first half of data
temp2<-training[-index,]  #Second half of data

#Removing NAs from temp1, turning char data into numeric and changing df to matrix
y=temp1$pstr        
X=temp1[,3:84]      
X$pstr=NULL         
X$SEX=as.numeric(as.factor(X$SEX)) 
X$higheduc=as.numeric(as.factor(X$higheduc)) 
# pstr_table=temp1%>%group_by(higheduc)%>%summarise(medianpstr=median(pstr)) 
# X_join=right_join(X,pstr_table,by="higheduc") 
#X_join$higheduc=NULL
X_join=as.matrix(X) 
X_join[,1]=(X_join[,1]-mean(X_join[,1]))/sd(X_join[,1])


immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
newdata=data.frame(apply(X_join,2,immedian))
newdata<-as.matrix(newdata)

#newdata<-scale(newdata) #scaling for temp1
#SVMs classifier on temp1
classifier = svm(formula = y ~ .,
                 data = newdata,
                 type = 'C-classification',
                 kernel = 'linear')

#Cleaning Data for temp2 just like for temp1
b=temp2$pstr        
a=temp2[,3:84]      
a$pstr=NULL         
a$SEX=as.numeric(as.factor(a$SEX)) 
a$higheduc=as.numeric(as.factor(a$higheduc)) 
# pstr_table2=temp2%>%group_by(higheduc)%>%summarise(medianpstr=median(pstr)) 
# X_join2=right_join(a,pstr_table2,by="higheduc") 
#table(X_join2$medianpstr)
#X_join2$higheduc=NULL
X_join2=as.matrix(a) 
X_join2[,1]=(X_join2[,1]-mean(X_join2[,1]))/sd(X_join2[,1])


immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
newdata2=data.frame(apply(X_join2,2,immedian))
newdata2<-as.matrix(newdata2)

#newdata2<-scale(newdata2)


# Predicting the temp2 set results
y_pred = predict(classifier, newdata2)
y_pred<-as.numeric(y_pred)
#b<-scale(b)
#y_pred<-scale(y_pred)
library(Metrics)
rmse(b,y_pred)


#predicting on test data
test_mat=testing[,3:84]      
test_mat$SEX=as.numeric(as.factor(test_mat$SEX))  
test_mat$higheduc=as.numeric(as.factor(test_mat$higheduc))
X_join_test=as.matrix(test_mat) 
X_join_test[,1]=(X_join_test[,1]-mean(X_join_test[,1]))/sd(X_join_test[,1])


immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
newdata_test=data.frame(apply(X_join_test,2,immedian))
newdata_test<-as.matrix(newdata_test)

#newdata_test<-scale(newdata_test)


testing$pstr<-predict(classifier, newdata_test)
testing$pstr<-as.numeric(testing$pstr)
#testing$pstr<-scale(testing$pstr)
write.csv(testing[,c("test_id","pstr")],"my_submission_svm.csv",row.names=F)

