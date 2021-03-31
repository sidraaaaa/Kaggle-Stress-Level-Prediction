#setwd("C:/Users/sidra/Desktop/IUPUI Senior Year/CSCI 495 Capstone")
training<-read.csv("./train.csv")
testing<-read.csv("./test.csv")
index=sample(nrow(training),nrow(training)/2,F)
temp1<-training[index,]
temp2<-training[-index,]

y=temp1$pstr         #holding out y label
X=temp1[,c("SEX","race6","hisp","hincome","pamarital",
             "fam_wage_loss_cv","think_will_hospitalized_cv","ext_fam_diag_cv",
             "fam_actions_cv___4","fam_actions_cv___10","fam_discord_cv","child_avg_elec_time_cv",
             "child_past_risk_cv___4","child_social_media_time_2_cv",
             "child_social_media_time_cv",
             "physical_activities_hr_cv","sitting_weekday_hour_cv","walk_10_min_per_day_cv")]  # for glmnet's numeric matrix
X$SEX=as.numeric(as.factor(X$SEX))
X=sapply(
  X,  function(i) ifelse(is.na(i), median(i, na.rm = TRUE),i)) #imputing NAs with median
X=as.matrix(X)

NN_keras_model<-keras_model_sequential()
NN_keras_model%>%
  layer_dense(units=50,activation='relu',
              input_shape = c(18))%>% #input shape: dim(X)= 4261 x 18
  layer_dropout(rate=0.4)%>%
  layer_dense(units=50,activation='relu')%>%
  layer_dropout(rate=0.4)%>%
  layer_dense(units=1,activation='linear')


summary(NN_keras_model)
NN_keras_model%>%compile(
  loss='MeanSquaredError',
  optimizer=optimizer_adam(),
  metrics=c('MeanSquaredError')
)  
history_X<-NN_keras_model%>%fit(X,y,epochs=30, batch_size=128,
                                     validation_split=0.2)    

plot(history_X)
pred<-NN_keras_model%>%predict(X)
rmse(y,pred)    #rmse 2.80



##########Other half of the data

n=temp2$pstr
m=temp2[,c("SEX","race6","hisp","hincome","pamarital",
           "fam_wage_loss_cv","think_will_hospitalized_cv","ext_fam_diag_cv",
           "fam_actions_cv___4","fam_actions_cv___10","fam_discord_cv","child_avg_elec_time_cv",
           "child_past_risk_cv___4","child_social_media_time_2_cv",
           "child_social_media_time_cv",
           "physical_activities_hr_cv","sitting_weekday_hour_cv","walk_10_min_per_day_cv")]  # for glmnet's numeric matrix
m$SEX=as.numeric(as.factor(m$SEX))
m=sapply(
  m,  function(i) ifelse(is.na(i), median(i, na.rm = TRUE),i)) #imputing NAs with median
m=as.matrix(m)

# NN_keras_model2<-keras_model_sequential()
# NN_keras_model2%>%
#   layer_dense(units=50,activation='relu',
#               input_shape = c(18))%>% #input shape: dim(X)= 4261 x 18
#   layer_dropout(rate=0.4)%>%
#   layer_dense(units=50,activation='relu')%>%
#   layer_dropout(rate=0.4)%>%
#   layer_dense(units=1,activation='linear')
# 
# 
# summary(NN_keras_model2)
# NN_keras_model2%>%compile(
#   loss='MeanSquaredError',
#   optimizer=optimizer_adam(),
#   metrics=c('MeanSquaredError')
# )  
history_m<-NN_keras_model%>%fit(m,n,epochs=30, batch_size=128,
                                validation_split=0.2)     #rmse= 2.77

plot(history_m)
pred2<-NN_keras_model%>%predict(m)
rmse(n,pred2)

######Working on Test data
#n=temp2$pstr
a=testing[,c("SEX","race6","hisp","hincome","pamarital",
           "fam_wage_loss_cv","think_will_hospitalized_cv","ext_fam_diag_cv",
           "fam_actions_cv___4","fam_actions_cv___10","fam_discord_cv","child_avg_elec_time_cv",
           "child_past_risk_cv___4","child_social_media_time_2_cv",
           "child_social_media_time_cv",
           "physical_activities_hr_cv","sitting_weekday_hour_cv","walk_10_min_per_day_cv")]  # for glmnet's numeric matrix
a$SEX=as.numeric(as.factor(a$SEX))
a=sapply(
  a,  function(i) ifelse(is.na(i), median(i, na.rm = TRUE),i)) #imputing NAs with median
#a=as.matrix(m)

#history_a<-NN_keras_model%>%fit(a,epochs=30, batch_size=128,
                                #validation_split=0.2)     #rmse= 2.77

#plot(history_a)
testing$pstr<-NN_keras_model%>%predict(a)
write.csv(testing[,c("test_id","pstr")],"my_submission_NN_Keras_glmnet_19.csv",row.names=F)
#rmse(n,pred2) #2.78
