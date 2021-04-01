#PCA

library(stats)
training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")
index=sample(nrow(training),nrow(training)/2,F)
temp1<-training[index,]
temp2<-training[-index,]

#First half of the data temp1
##########Feature Scaling columns
y=temp1$pstr         #holding out y label
X=temp1[,3:84]      # for glmnet's numeric matrix
X$pstr=NULL         #removing pstr
X$SEX=as.numeric(as.factor(X$SEX))   #factoring the categorical value

X_join$higheduc=NULL
#x.imp<-missForest(X_join,maxiter=4,verbose = TRUE) #trying missForest to impute NAs
immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
newdata=data.frame(apply(X_join,2,immedian))

x.imp<-as.matrix(x.imp)


#########PCA
library(FactoMineR) #Library for PCA
X_pca<-PCA(X_join)#applying PCA to our numeric matrix
X_pca$eig #seeing eigen values for our PCA
X_pca$var$cos2 #seeing cos2 values for our PCA
X_pca$var$contrib #seeing contrib values for our PCA
dimdesc(X_pca) 
print(X_pca)
library("factoextra")
eig.val <- get_eigenvalue(X_pca)
fviz_eig(X_pca, addlabels = TRUE, ylim = c(0, 81)) #Scree plot to see which features are good
#35% variances is retained by first seven pcas and 20% variance by first two PCA
fviz_pca_var(X_pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE
) #The more closer to the circumference of the circle the more qualitative
#Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true") #for installation
#install_github("vqv/ggbiplot")
fviz_eig(X_pca) #visualizing eigen values


#Data Prep for KMEANS Clustering
x.imp<-as.numeric(x.imp)
X_scaled<-scale(x.imp)
#Number of clusters
set.seed(123)
km.res <- kmeans(x.imp, 4, nstart = 25)


#plot(X_pca, y,which="screeplot",classic=TRUE, cutoff=0.20)

# Backward Feature selection
#using olsrr package for backward selection
model <- lm(y ~ ., data = newdata)
j <- ols_step_forward_p(model)  #Eliminating features based on p-values
k <- ols_step_backward_p(model) #Eliminating features based on p-values
library(MASS)
stepAIC(model, direction = "both", trace = FALSE) #using stepAIC with both forward and backward direction

###Using glm with stepAIC (both selected variables)
stepAICmodel<-glm(formula = y ~ interview_age + SEX + ageyear + pamarriedliving + 
     race6 + hisp + raceeth5 + hincome + pamarital + fam_size_cv + 
     fam_actions_cv___3 + fam_actions_cv___10 + fam_actions_cv___15 + 
     fam_discord_cv + fam_exp6_cv + su_p_other_drugs_cv + avoid_talking_about_cv + 
     child_avg_elec_min_cv + child_avg_elec_time_2_cv + child_avg_elec_time_cv + 
     child_gaming_time_cv + child_past_risk_cv___3 + child_past_risk_cv___4 + 
     child_phys_activity_min_cv + child_social_media_min_cv + 
     child_social_media_time_cv + physical_activities_hr_cv + 
     sitting_weekday_hour_cv + walk_10_min_per_day_cv, data = newdata)
y2=temp2$pstr         #holding out y label
X2=temp2[,3:84]      # for glmnet's numeric matrix
X2$pstr=NULL         #removing pstr
X2$SEX=as.numeric(as.factor(X2$SEX))   #factoring the categorical value

X2$higheduc=NULL
#x.imp<-missForest(X_join,maxiter=4,verbose = TRUE) #trying missForest to impute NAs
immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
newdata2=data.frame(apply(X2,2,immedian))


pred<-predict(stepAICmodel,newdata2,type="response")
library(Metrics)
rmse(temp2$pstr,pred) #predicting rmse 2.80

#working on Test data
t2=testing[,3:84]      # for glmnet's numeric matrix
t2$SEX=as.numeric(as.factor(t2$SEX))   #factoring the categorical value
t2$higheduc=NULL

#x.imp<-missForest(X_join,maxiter=4,verbose = TRUE) #trying missForest to impute NAs
immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
newdata_t=data.frame(apply(t2,2,immedian))

testing$pstr=predict(stepAICmodel,newdata_t,type="response")
testing$pstr<-round(testing$pstr, digits = 0)
write.csv(testing[,c("test_id","pstr")],"my_submission_stepAICmodel.csv",row.names=F)
