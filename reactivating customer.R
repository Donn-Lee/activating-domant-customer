library(data.table)
library(sqldf)
library(doMC)
library(ROCR)
library(glmnet)
library(caret)
library(dplyr)
#取消科学计数法表示
options(scipen=999)
path ='C:/Users/you are the worst/Desktop/Columbia msor/BA501/first_project'
setwd(path)
################################################
#### Step 0, read in data and do some basic cleaning
################################################

customer_table <- fread("customer_table.csv")
order_table <- fread("order_table.csv")
product_table <- fread("product_table.csv")
category_table <- fread("category_table.csv")

#### to change from scientific notation to the actual number
customer_table[,customer_id := as.character(customer_table$customer_id)]
order_table[,customer_id := as.character(order_table$customer_id)]
order_table[,order_id := as.character(order_table$order_id)]
order_table[,product_id := as.character(order_table$product_id)]
product_table[,product_id := as.character(product_table$product_id)]

################################################
#### Step 1, get our target audience, with the training flag (dependent variable)
################################################

## find customers who only made one purchase before 2016/12/22
base <- subset( 
    order_table[
       order_date<'20161222'&order_amount>0,
       .(count=.N, order_date=max(order_date), order_amount=max(order_amount),product_id=max(product_id)),
      by='customer_id'
    ],
    count==1)

#subset 语法subset(x, subset（相当于where）, select（挑选哪几列）, drop = FALSE, …)

## find customers who made purchase between 2016/12/22 and 2017/02/22
purchase_again <- sqldf("SELECT customer_id
                        , MAX(order_date) AS latest_orderdate
                        FROM order_table
                        WHERE order_date BETWEEN '20161222' and '20170222'
                        GROUP BY 1");

## find customers who were dormant between 2016/12/22 and 2017/02/22
dormant_3month <- sqldf("SELECT * 
                        FROM base 
                        WHERE customer_id NOT IN 
                        (SELECT customer_id FROM purchase_again);")

## find customers who purchased again between 2017/02/23 and 2017/05/22
purchase_flag <- sqldf("SELECT customer_id
                       FROM order_table
                       WHERE order_date BETWEEN '20170223' and '20170522'
                       GROUP BY 1");

is_converted <- sqldf("SELECT CASE WHEN pf.customer_id IS NOT NULL THEN 1
                      ELSE 0 END AS is_converted 
                      FROM dormant_3month d
                      LEFT JOIN purchase_flag pf
                      ON d.customer_id = pf.customer_id
                      ;")

## pull all features for the targeted audience
user_cohort <- sqldf("SELECT * 
                     FROM customer_table
                     WHERE customer_id IN (SELECT customer_id FROM dormant_3month)")

## combine the dependent variable with independent variables
user_features <- data.table(
         cbind(is_converted,user_cohort,dormant_3month[,c("order_date","order_amount","product_id")])
         )

## remove temporary tables
rm(base,dormant_3month,is_converted,purchase_again,purchase_flag,user_cohort)
gc()

################################################
#### Step 2, data pre-processing
################################################

## remove ML unrelated field
user_features[,c("customer_id","last_visit_date","age") := NULL]
#age是均匀分布，没有必要

## calculate new features, how long has this customer been around?
user_features[,tenure := as.numeric(as.Date("2017-02-22") - 
                                      as.Date(first_visit_date))]
user_features[,days_since_first_order := as.numeric(as.Date("2017-02-22") -
                                                      as.Date(as.character(order_date),format='%Y%m%d'))]
#这里的format是as.character(order_date)的格式‘年月日’



user_features <- data.table(
  sqldf(" 
        SELECT uf.*
        , CASE WHEN pt.category_id IS NOT NULL THEN pt.category_id ELSE 'unknown' END AS category_id
        FROM user_features uf
        LEFT JOIN product_table pt
        ON uf.product_id = pt.product_id;"))

user_features[,c("first_visit_date","order_date","product_id") := NULL]

## deal with missing value: fill all NA's with 0
user_features$category_id[is.na(user_features$category_id)] <- 'unknown'
user_features$latest_device_class[is.na(user_features$latest_device_class)] <- 'unknown'
user_features[is.na(user_features)] <- 0

## find out top 20 countries
top_countries <- user_features[,list(Count=.N), by = country][order(-Count)][1:30][,country]

user_features[,country_updated := ifelse(country %in% top_countries,country,'others')]
user_features[,c("country") := NULL]

## dummy variables
country_dummy <- model.matrix( ~ country_updated - 1, data=user_features)
#data.table(class.ind(user_features$country_updated))
gender_dummy <- model.matrix( ~ gender - 1, data=user_features)
category_dummy <- model.matrix( ~ category_id - 1, data=user_features)
deviceclass_dummy <- model.matrix( ~ latest_device_class - 1, data=user_features)

user_features_combined <- cbind(user_features,country_dummy,gender_dummy,category_dummy,deviceclass_dummy)
user_features_combined[,c("country_updated","gender","category_id","latest_device_class") := NULL]
rm(country_dummy,gender_dummy,category_dummy,deviceclass_dummy)

## zero variance variable
zero_variance <- names(Filter(function(x)(length(unique(x)) == 1),user_features_combined))
#某列值都一样，这种不能做回归

## filter out zero variance variable
user_features_combined <- Filter(function(x)(length(unique(x)) > 1),user_features_combined)

## transform continuous variables
transform_columns <- c("user_feature","phone_feature","tablet_feature","family_size","number_of_devices","tenure","order_amount","days_since_first_order")
transformed_column     <- user_features_combined[ ,grepl(paste(transform_columns, collapse = "|"),names(user_features_combined)),with = FALSE]
non_transformed_column <- user_features_combined[ ,-grepl(paste(transform_columns, collapse = "|"),names(user_features_combined)),with = FALSE]
transformed_column_processed <- predict(preProcess(transformed_column, method = c("BoxCox","scale")),transformed_column)
#数据标准化



transformedAll <- cbind(non_transformed_column, transformed_column_processed)
rm(customer_table,order_table,product_table,user_features,transformed_column_processed,transformed_column,non_transformed_column)

################################################
#### Step 3, set up training/test set
################################################


set.seed(1003)

train_rate <- 0.6
training_index <- createDataPartition(transformedAll$is_converted, p = train_rate, list = FALSE, times = 1)

train_data <- transformedAll[training_index,]
test_data <- transformedAll[-training_index,]

train_x <- subset(train_data, select = -c(is_converted))
train_y <- as.factor(apply(subset(train_data, select = c(is_converted)), 2, as.factor))
train_y_categorial <- ifelse(train_y == 1, "YES", "NO")

test_x <- subset(test_data, select = -c(is_converted))
test_y <- as.factor(apply(subset(test_data, select = c(is_converted)), 2, as.factor))

################################################
#### Step 4, train a model
################################################

registerDoMC(cores=6)

#### Lasso Logistic Regression
model_glm_cv_lasso <- cv.glmnet(data.matrix(train_x),train_y_categorial,alpha = 1,family="binomial",type.measure="auc",parallel=TRUE)

#### Ridge Logistic Regression
model_glm_cv_ridge <- cv.glmnet(data.matrix(train_x),train_y_categorial,alpha = 0,family="binomial",type.measure="auc",parallel=TRUE)

#### Random Forest
rf <- foreach(ntree=rep(200, 6), .combine=combine, .multicombine=TRUE,
              .packages='randomForest') %dopar% {
                randomForest(train_x, train_y, ntree=ntree)
              }

## Try KNN, Neural Net

################################################
#### Step 5, evaluate the model performance 
################################################
lasso_predict <- predict(model_glm_cv_lasso, data.matrix(test_x),type='response')

lasso_pred <- prediction(lasso_predict,test_y)
lasso_perf_recall <- performance(lasso_pred,"prec","rec")
lasso_perf_roc <- performance(lasso_pred,"tpr","fpr")
lasso_perf_auc <- performance(lasso_pred,"auc")

################################################
#### What is next? 
################################################

#### Demo presentation for phase 3
## 1. explain how to use the model for direct targeting (and why we should use it?)
## 2. Do the analysis by analyzing why we should target users who have been dormant for three months. 
###The idea is that run analysis for repeat buyers, how much they spend within half years after their second purchase, breakdown by the days difference between first purchase and second purchase.
## 3. build the slides (from the business solution to your final recommendation)

repeat_buyers <- subset(
  order_table[order_amount>0,
              .(count=.N),by=customer_id],
  count>1)

orders_sub <- sqldf("SELECT * FROM order_table WHERE customer_id IN (SELECT customer_id FROM repeat_buyers GROUP BY 1)");

orders_sub$order_date <- as.Date(as.character(orders_sub$order_date),format='%Y%m%d')

orders_sub <- orders_sub[order(orders_sub$customer_id),]
orders_sub$order_rank <- unlist(with(orders_sub, tapply(order_date, customer_id, function(x) rank(x, ties.method= "first"))))
orders_sub$customer_id <- as.character(orders_sub$customer_id)

first_order <- subset(orders_sub[,c("customer_id","order_date","order_rank")],order_rank==1)
second_order <- subset(orders_sub[,c("customer_id","order_date","order_rank")],order_rank==2)

f_to_s <- merge(first_order,second_order,by='customer_id')

f_to_s$f2s_days <- as.numeric(f_to_s$order_date.y - f_to_s$order_date.x)
f_to_s <- data.table(f_to_s)


orders_after2 <- subset(orders_sub,order_rank>=3)

#find distribution for X-month dormancy
domant_customer <- f_to_s[f2s_days < 150]
#domant_customer$order_date <- as.Date(as.character(domant_customer$order_date),format='%Y%m%d')


#second_order$order_date <- as.Date(as.character(second_order$order_date),format='%Y%m%d')


second_order_2 <- sqldf("select so.* from second_order so inner join domant_customer dc
                        on so.customer_id = dc.customer_id")


user_stats <- sqldf("SELECT so.customer_id, SUM(oa.order_amount) AS sum_rev 
                    FROM second_order_2 so INNER JOIN orders_after2 oa ON so.customer_id = oa.customer_id
                    WHERE oa.order_date - so.order_date <= 183 GROUP BY 1;")

mean(user_stats$sum_rev)
