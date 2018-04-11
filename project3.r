rm(list=ls());
# install.packages(c("randomForest", "caret", "ROCR", "DMwR", "data.table", "zoo"))

library(data.table)
library(randomForest)
library(caret)
library(ROCR)
library(DMwR)
library(zoo)
library(e1071)

df <- fread("~/Desktop/BA501/project3/creditcard.csv")

#### Exploratory analysis
summary(df)
sum(is.na(df))

ggplot(df, aes(x=V3)) + geom_density(aes(group=Class, colour=Class, fill=Class), alpha=0.3)

#### Data pre-processing
## nothing

#### Training and Test dataset
df[,Class:=as.factor(Class)]

set.seed(1003)

training_index <- createDataPartition(df$Class, p=0.7,list=FALSE)
training <- df[training_index,]
test<- df[-training_index,]















### Logistic regression
logit <- glm(Class ~ ., data = training, family = "binomial")
logit_pred <- predict(logit, test, type = "response")

logit_prediction <- prediction(logit_pred,test$Class)
logit_recall <- performance(logit_prediction,"prec","rec")
logit_roc <- performance(logit_prediction,"tpr","fpr")
logit_auc <- performance(logit_prediction,"auc")

### Random forest
rf.model <- randomForest(Class ~ ., data = training,ntree = 500, nodesize = 20)
rf_pred <- predict(rf.model, test,type="prob")

rf_prediction <- prediction(rf_pred[,2],test$Class)
rf_recall <- performance(rf_prediction,"prec","rec")
rf_roc <- performance(rf_prediction,"tpr","fpr")
rf_auc <- performance(rf_prediction,"auc")

### Bagging Trees
ctrl <- trainControl(method = "cv", number = 10)

tb_model <- train(Class ~ ., data = train_smote, method = "treebag",
                 trControl = ctrl)

tb_pred <- predict(tb_model$finalModel, test, type = "prob")

tb_prediction <- prediction(tb_pred[,2],test$Class)
tb_recall <- performance(logit_prediction,"prec","rec")
tb_roc <- performance(logit_prediction,"tpr","fpr")
tb_auc <- performance(logit_prediction,"auc")

plot(logit_recall,col='red')
plot(rf_recall, add = TRUE, col = 'blue')
plot(tb_recall, add = TRUE, col = 'green')

#### Functions to calculate 'area under the pr curve'
auprc <- function(pr_curve) {
 x <- as.numeric(unlist(pr_curve@x.values))
 y <- as.numeric(unlist(pr_curve@y.values))
 y[is.nan(y)] <- 1
 id <- order(x)
 result <- sum(diff(x[id])*rollmean(y[id],2))
 return(result)
}

auprc_results <- data.frame(logit=auprc(logit_recall)
                            , rf = auprc(rf_recall)
                            , tb = auprc(tb_recall))
