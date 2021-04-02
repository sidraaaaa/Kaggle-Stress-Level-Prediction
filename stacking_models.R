#Stacking different models together

training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")

index <- sample(2,nrow(training),replace=TRUE,prob=c(0.5,0.5)) #Splitting data into training and testing data
training[is.na(training)]=-1
testing[is.na(testing)]=-1


#Removing NAs in training data
train=training[,3:84]      # choosing variables
train$SEX=as.numeric(as.factor(train$SEX))   #factoring the categorical value

train$higheduc=as.numeric(as.factor(train$higheduc))
immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
train=data.frame(apply(train,2,immedian)) #using above function to impute median->Dataframe

train_mat<-as.matrix(train) #Matrix of train


#Removing NAs in testing data
test=testing[,3:84]      # choosing variables
test$SEX=as.numeric(as.factor(test$SEX))   #factoring the categorical value

test$higheduc=as.numeric(as.factor(test$higheduc))
immedian=function(x){
  x<-as.numeric(as.character(x)) #first convert each column into numeric if it is from factor
  x[is.na(x)] =median(x, na.rm=TRUE) #convert the item with NA to median value from the column
  x #display the column
}
test=data.frame(apply(test,2,immedian)) #using above function to impute median->Dataframe

test_mat<-as.matrix(test) #Matrix of test



#Splitting train df to temp1 and temp2 for cross validation set
index=sample(nrow(train),nrow(train)/2,F)
temp1<-train[index,] #first half of data
temp2<-train[-index,]  #Second half of data

#Applying glm on randomly selected features
model_glm<-glm(pstr~SEX+ageyear+higheduc+pamarriedliving+race6+hisp+raceeth7+hincome+pamarital6+fam_size_cv+
                 fam_wage_loss_cv+su_p_alc_drinks_used_cv+su_p_alc_use_cv+su_p_cig_use_cv+su_p_mj_smoke_cv+worry_about_cv+
                 ext_fam_diag_cv+fam_exp1_cv+think_will_hospitalized_cv+fam_actions_cv___1+fam_actions_cv___14+fam_discord_cv+
                 fam_exp5_cv+fam_exp6_cv+su_p_other_drugs_cv+su_p_rx_od_cv+absent_reason_cv___3+child_past_risk_cv___2+
                 child_phys_activity_days_cv+walk_10_min_per_day_cv+child_video_chat_min_cv+child_texting_min_cv+
                 child_social_media_min_cv,data=temp1,family=gaussian)

temp2$glm_prediction<-predict(model_glm,temp2,type="response") #storing results in cross validated set's Feature
library(Metrics)
rmse(temp2$pstr,temp2$glm_prediction)#2.834 ///No need of rounding off here
test$glm_prediction<-predict(model_glm,test,type="response") #storing prediction in test


#Top 11 Features from GLMNET and combining with GLM
model_glmnet_glm<-glm(pstr~SEX+hisp+hincome+fam_actions_cv___4+fam_discord_cv+
                        child_avg_elec_time_cv+child_social_media_time_cv+
                        physical_activities_hr_cv+sitting_weekday_hour_cv+
                        walk_10_min_per_day_cv,data=temp1,family=gaussian)

model_glm2<-glm(pstr~interview_age + SEX + ageyear + pamarriedliving + 
                  race6 + hisp + raceeth5 + hincome + pamarital + fam_size_cv + 
                  fam_actions_cv___3 + fam_actions_cv___10 + fam_actions_cv___15 + 
                  fam_discord_cv + fam_exp6_cv + su_p_other_drugs_cv + avoid_talking_about_cv + 
                  child_avg_elec_min_cv + child_avg_elec_time_2_cv + child_avg_elec_time_cv + 
                  child_gaming_time_cv + child_past_risk_cv___3 + child_past_risk_cv___4 + 
                  child_phys_activity_min_cv + child_social_media_min_cv + 
                  child_social_media_time_cv + physical_activities_hr_cv + 
                  sitting_weekday_hour_cv + walk_10_min_per_day_cv,data=temp1,family=gaussian)

temp2$glmnettop_prediction<-predict(model_glmnet_glm,temp2,type="response")
rmse(temp2$pstr,temp2$glmnettop_prediction)#RMSE 2.814
test$glmnettop_prediction<-predict(model_glmnet_glm,test,type="response") #storing prediction in test

# pd1<-predict(model_glm2,temp2,type="response")
# rmse(temp2$pstr,pd1)#RMSE 2.8172

#Top 11 Features from GLMNET and NN
y1=temp1$pstr         #holding out y1 label and X1 matrix for temp1 
X1=temp1[,c("SEX","race6","hisp","hincome","pamarital",
           "fam_wage_loss_cv","think_will_hospitalized_cv","ext_fam_diag_cv",
           "fam_actions_cv___4","fam_actions_cv___10","fam_discord_cv","child_avg_elec_time_cv",
           "child_past_risk_cv___4","child_social_media_time_2_cv",
           "child_social_media_time_cv",
           "physical_activities_hr_cv","sitting_weekday_hour_cv","walk_10_min_per_day_cv")]  # for glmnet's numeric matrix

X1=as.matrix(X1)

y2=temp2$pstr   #holding out y2 label and X2 matrix for temp2 
X2=temp2[,c("SEX","race6","hisp","hincome","pamarital",
            "fam_wage_loss_cv","think_will_hospitalized_cv","ext_fam_diag_cv",
            "fam_actions_cv___4","fam_actions_cv___10","fam_discord_cv","child_avg_elec_time_cv",
            "child_past_risk_cv___4","child_social_media_time_2_cv",
            "child_social_media_time_cv",
            "physical_activities_hr_cv","sitting_weekday_hour_cv","walk_10_min_per_day_cv")]  # for glmnet's numeric matrix

X2=as.matrix(X2)

t=test[,c("SEX","race6","hisp","hincome","pamarital",
            "fam_wage_loss_cv","think_will_hospitalized_cv","ext_fam_diag_cv",
            "fam_actions_cv___4","fam_actions_cv___10","fam_discord_cv","child_avg_elec_time_cv",
            "child_past_risk_cv___4","child_social_media_time_2_cv",
            "child_social_media_time_cv",
            "physical_activities_hr_cv","sitting_weekday_hour_cv","walk_10_min_per_day_cv")]  # for glmnet's numeric matrix

t=as.matrix(t)

install.packages("devtools")
library(devtools)
install.packages("keras")
devtools::install_github("rstudio/keras")
library(keras)

NN_keras_model<-keras_model_sequential()
NN_keras_model%>%
  layer_dense(units=50,activation='relu',
              input_shape = c(18))%>% #input shape: dim(X)= 4261 x 18
  layer_dropout(rate=0.05)%>%
  layer_dense(units=50,activation='relu')%>%
  layer_dropout(rate=0.05)%>%
  layer_dense(units=1,activation='linear')


summary(NN_keras_model)
NN_keras_model%>%compile(
  loss='MeanSquaredError',
  optimizer=optimizer_adam(),
  metrics=c('MeanSquaredError')
)  
history_X<-NN_keras_model%>%fit(X1,y1,epochs=30, batch_size=128,
                                validation_split=0.2)    
nn_prediction<-NN_keras_model%>%predict(X1) #Predicting on temp1's matrix
rmse(y1,nn_prediction) #calculating rmse 2.744

temp2$nn_prediction<-NN_keras_model%>%predict(X2) #Predicting on temp2's matrix
rmse(y2,temp2$nn_prediction)#calculating rmse 2.815
test$nn_prediction<-NN_keras_model%>%predict(t)


#GBM
library(gbm)

model_gbm<-gbm(pstr~SEX+hisp+hincome+fam_actions_cv___4+fam_discord_cv+
                child_avg_elec_time_cv+child_social_media_time_cv+
                physical_activities_hr_cv+sitting_weekday_hour_cv+
                walk_10_min_per_day_cv,data = temp1, n.trees = 150)

temp2$gbm_prediction<-predict(model_gbm,newdata=temp2,n.trees=150,type="response")
rmse(temp2$pstr,temp2$gbm_prediction)#rmse 2.811
test$gbm_prediction<-predict(model_gbm,newdata=test,n.trees=150,type="response")



############################
###############################Random Forest
library(randomForest)

model_rf2<-randomForest(pstr~interview_age + SEX + ageyear + pamarriedliving + 
                          race6 + hisp + raceeth5 + hincome + pamarital + fam_size_cv + 
                          fam_actions_cv___3 + fam_actions_cv___10 + fam_actions_cv___15 + 
                          fam_discord_cv + fam_exp6_cv + su_p_other_drugs_cv + avoid_talking_about_cv + 
                          child_avg_elec_min_cv + child_avg_elec_time_2_cv + child_avg_elec_time_cv + 
                          child_gaming_time_cv + child_past_risk_cv___3 + child_past_risk_cv___4 + 
                          child_phys_activity_min_cv + child_social_media_min_cv + 
                          child_social_media_time_cv + physical_activities_hr_cv + 
                          sitting_weekday_hour_cv + walk_10_min_per_day_cv, nTree=1000, data=temp1)


temp2$rf_prediction<-predict(model_rf2,newdata=temp2,n.trees=1000,type="response")
rmse(temp2$pstr,temp2$rf_prediction)#rmse 2.797
test$rf_prediction<-predict(model_rf2,newdata=test,n.trees=1000,type="response")


#plotting the model predictions
plot(temp2$glm_prediction,temp2$glmnettop_prediction)
cor(temp2$glm_prediction,temp2$glmnettop_prediction)
     #temp2$nn_prediction,temp2$gbm_prediction,temp2$rf_prediction)


#Stacking models
model_stack<-glm(pstr~SEX*glm_prediction+ageyear*glm_prediction+higheduc*glm_prediction
                +pamarriedliving*glm_prediction+race6*glm_prediction+hisp*glm_prediction
                +raceeth7*glm_prediction+hincome*glm_prediction+pamarital6*glm_prediction
                +fam_size_cv*glm_prediction+fam_wage_loss_cv*glm_prediction
                +su_p_alc_drinks_used_cv*glm_prediction+su_p_alc_use_cv*glm_prediction
                +su_p_cig_use_cv*glm_prediction+su_p_mj_smoke_cv*glm_prediction
                +worry_about_cv*glm_prediction+ext_fam_diag_cv*glm_prediction
                +fam_exp1_cv*glm_prediction+think_will_hospitalized_cv*glm_prediction
                +fam_actions_cv___1*glm_prediction+fam_actions_cv___14*glm_prediction
                +fam_discord_cv*glm_prediction+fam_exp5_cv*glm_prediction
                +fam_exp6_cv*glm_prediction+su_p_other_drugs_cv*glm_prediction
                +su_p_rx_od_cv*glm_prediction+absent_reason_cv___3*glm_prediction
                +child_past_risk_cv___2*glm_prediction
                +child_phys_activity_days_cv*glm_prediction
                +walk_10_min_per_day_cv*glm_prediction+child_video_chat_min_cv*glm_prediction
                +child_texting_min_cv*glm_prediction+child_social_media_min_cv*glm_prediction
                
                +SEX*glmnettop_prediction+hisp*glmnettop_prediction+hincome*glmnettop_prediction
                +fam_actions_cv___4*glmnettop_prediction+fam_discord_cv*glmnettop_prediction
                +child_avg_elec_time_cv*glmnettop_prediction+child_social_media_time_cv*glmnettop_prediction
                +physical_activities_hr_cv*glmnettop_prediction
                +sitting_weekday_hour_cv*glmnettop_prediction+walk_10_min_per_day_cv*glmnettop_prediction
                
                +SEX*nn_prediction+race6*nn_prediction+hisp*nn_prediction
                +hincome*nn_prediction+pamarital*nn_prediction+fam_wage_loss_cv*nn_prediction
                +think_will_hospitalized_cv*nn_prediction+ext_fam_diag_cv*nn_prediction
                +fam_actions_cv___4*nn_prediction+fam_actions_cv___10*nn_prediction
                +fam_discord_cv*nn_prediction+child_avg_elec_time_cv*nn_prediction
                +child_past_risk_cv___4*nn_prediction+child_social_media_time_2_cv*nn_prediction
                +child_social_media_time_cv*nn_prediction+physical_activities_hr_cv*nn_prediction
                +sitting_weekday_hour_cv*nn_prediction+walk_10_min_per_day_cv*nn_prediction
                
                +SEX*gbm_prediction+hisp*gbm_prediction+hincome*gbm_prediction
                +fam_actions_cv___4*gbm_prediction+fam_discord_cv*gbm_prediction
                +child_avg_elec_time_cv*gbm_prediction+child_social_media_time_cv*gbm_prediction
                +physical_activities_hr_cv*gbm_prediction
                +sitting_weekday_hour_cv*gbm_prediction+walk_10_min_per_day_cv*gbm_prediction
                
                +interview_age*rf_prediction + SEX*rf_prediction + ageyear*rf_prediction 
                + pamarriedliving*rf_prediction + race6*rf_prediction + hisp*rf_prediction 
                + raceeth5*rf_prediction + hincome*rf_prediction + pamarital*rf_prediction 
                + fam_size_cv*rf_prediction + fam_actions_cv___3*rf_prediction 
                + fam_actions_cv___10*rf_prediction + fam_actions_cv___15*rf_prediction 
                + fam_discord_cv*rf_prediction + fam_exp6_cv*rf_prediction 
                + su_p_other_drugs_cv*rf_prediction + avoid_talking_about_cv*rf_prediction 
                + child_avg_elec_min_cv*rf_prediction + child_avg_elec_time_2_cv*rf_prediction 
                + child_avg_elec_time_cv*rf_prediction + child_gaming_time_cv*rf_prediction 
                + child_past_risk_cv___3*rf_prediction + child_past_risk_cv___4*rf_prediction 
                + child_phys_activity_min_cv*rf_prediction + child_social_media_min_cv*rf_prediction 
                + child_social_media_time_cv*rf_prediction + physical_activities_hr_cv*rf_prediction 
                + sitting_weekday_hour_cv*rf_prediction + walk_10_min_per_day_cv*rf_prediction  
                  ,
                data=temp2)

testing$pstr=predict(model_stack,test,type="response")

write.csv(testing[,c("test_id","pstr")],"my_stack_prediction.csv",row.names=F)
