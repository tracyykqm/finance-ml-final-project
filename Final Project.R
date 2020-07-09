library(quantmod)
library(tidyquant)
library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(glmnet)
library(readr)
#library(recipes)
library(quanteda)


#################################################################
# Functions and import data
#################################################################

# Function to compute coefficient of determination (R2)
r2_score_fun <- function(outcome, predictor) {
  c(1 - var(outcome-predictor)/var(outcome))
}

# Function to compute MSE
mse_fun <- function(outcome, predictor) {
  mean(unlist((outcome - predictor)^2))
}



#################################################################
# Import and transfer data
#################################################################

full_data <- read_csv(file.choose())

#choose columns that we want to include
data <- subset(full_data, select = c(ticker, issuer,`factset analytics insights`,
                            `expense ratio`, `AUM(Millions $)`, 
                            `p/e`, `p/b`, `dist yld`,
                             `num holdings`, `weighting methodology`,
                            `3 yr perform`,`otc derivative use`,
                            `securities lending active`, `fund closure risk`,
                            `avg daily share volume`,
                            `avg spread %`,`avg spread dollar`,
                            `median premium/discount 12 mo`,
                            `creation unit/day 45 day avg`,
                             `Beta`, `up beta`, `down beta`)) %>%
  drop_na()


#convert characters with number into numeric class
data$`expense ratio` <- as.numeric(sub("%", "", data$`expense ratio`))/100
data$`dist yld` <- as.numeric(sub("%", "", data$`dist yld`))/100
data$`3 yr perform` <- as.numeric(sub("%", "", data$`3 yr perform`))/100
data$`p/e` <- as.numeric(data$`p/e`)
data$`p/b` <- as.numeric(data$`p/b`)
data$Beta <- as.numeric(data$`Beta`)
data$`up beta` <- as.numeric(data$`up beta`)
data$`down beta` <- as.numeric(data$`down beta`)
data <- data %>% drop_na()

# 
# adding transformation
data$`p/e squared` <- data$`p/e`^2
data$`p/b squared` <- data$`p/b`^2
data$`p/e * Beta` <- data$`p/e`*data$`p/b`
data$`p/b * Beta` <- data$`p/b`*data$Beta
data$`avg daily share volume squared` <- data$`avg daily share volume`^2
data$`num holdings squared` <- data$`num holdings`^2
data$`avg spread % squared` <- data$`avg spread %`^2



#rank etfs by 3 yr perform
data$`3yr perform rank` <- rank(-data$`3 yr perform`, ties.method = "min")

#if issuer is one of the big 5 issuers
big_5_issuers <- c("Blackrock", "Vanguard", "State Street Global Advisors",
                 "Invesco", "Charles Schwab	")
data$`big 5 issuer` <- ifelse (data$issuer %in% big_5_issuers, 1, 0)


#convert weighting methodology into binary
data$`market cap weighting method` <- 
  ifelse (data$`weighting methodology` == "Market Cap", 1, 0)

data$`market cap weighting method` <- 
  ifelse (data$`weighting methodology` == "Market Value", 1, 0)

#convert if fund has used derivative into binary
data$`derivative use` <- 
  ifelse (data$`otc derivative use` == "Yes", 1, 0)

#convert lending activity into binary 
data$`lending` <- 
  ifelse (data$`securities lending active` == "Yes", 1, 0)

#convert lending activity into binary 
data$`low fund closure risk` <- 
  ifelse (data$`fund closure risk` == "Low", 1, 0)


#################################################################
# text data processing
#################################################################

# data$`text length` <- nchar(data$`factset analytics insights`)
# #summary(data$`text length`)
# 
# # tokenize insights
# text_token <- tokens(data$`factset analytics insights`, what = "word",
#                      remove_numbers = TRUE, remove_punct = TRUE,
#                      remove_symbols = TRUE, split_hyphens = TRUE)
# 
# # lower case the tokens
# text_token <- tokens_tolower(text_token)
# 
# # remove stopwords
# text_token <- tokens_select(text_token, stopwords(),
#                             selection = "remove")
# # perform stemming
# text_token <- tokens_wordstem(text_token, language = "english")
# 
# # create bag-of-word model
# text_token_dfm <- dfm(text_token, tolower = FALSE)
# 
# data_tokens_df <- cbind(data, convert(text_token_dfm, to = "data.frame"))
# 

#create a new data frame without any characters in it
# data_reg <- subset(data_tokens_df, select = -c(`ticker`, `issuer`, 
#                                                `factset analytics insights`,
#                                                `weighting methodology`,
#                                      `otc derivative use`, `3 yr perform`,
#                                      `securities lending active`,
#                                      `fund closure risk`, `doc_id`))
                   
data_reg <- subset(data, select = -c(`ticker`, `issuer`,
                                               `factset analytics insights`,
                                               `weighting methodology`,
                                     `otc derivative use`, `3 yr perform`,
                                     `securities lending active`,
                                     `fund closure risk`))


#################################################################
# Build feature and label matrix
#################################################################

# Get training and test sets
train_set   <- data_reg[1:400, ]; rownames(train_set) <- NULL
verify_set  <- data_reg[401:620, ]; rownames(verify_set) <- NULL
test_set    <- data_reg[-c(1:620), ]; rownames(test_set) <- NULL

# Build target and features matrix
X_train   <- train_set %>% select(-c(`3yr perform rank`))
Y_train  <- train_set %>% select(`3yr perform rank`)


X_verify   <- verify_set %>% select(-c(`3yr perform rank`))
Y_verify  <- verify_set %>% select(`3yr perform rank`)


X_test   <- test_set %>% select(-c(`3yr perform rank`))
Y_test  <- test_set %>% select(`3yr perform rank`)



#################################################################
# Standardize non-binary features
#################################################################

# stadardize columns without binary variables
scaler <- preProcess(X_train[-c(22:26)], method=c("center", "scale"))
scaler$std <- scaler$std*sqrt((dim(X_train)[1]-1)/dim(X_train)[1])

X_tr_sd <- predict(scaler, X_train)
X_ve_sd <- predict(scaler, X_verify)
X_te_sd <- predict(scaler, X_test)



#################################################################
# Predict returns - 3yr perform 
#################################################################
# Lambda corresponds to alpha in Python code
lambda_0  <- 1.0
l1        <- 0.7

# OLS
regr <- glmnet(as.matrix(X_tr_sd), as.matrix(Y_train), alpha=0, 
               lambda=0, standardize=F)

# ridge
ridge <- glmnet(as.matrix(X_tr_sd), as.matrix(Y_train), alpha=0, 
                lambda=lambda_0*2/nrow(as.matrix(X_tr_sd)), standardize=F)

# Lasso
lasso <- glmnet(as.matrix(X_tr_sd), as.matrix(Y_train), alpha=1, 
                lambda=lambda_0, standardize=F)

# Elastic Net
enet <- glmnet(as.matrix(X_tr_sd), as.matrix(Y_train), alpha=l1, 
               lambda=lambda_0, standardize=F)

# Construct coefficients array
coeffs <- data.frame(Variable=rownames(coef(regr)), OLS=as.numeric(coef(regr)), 
                     RIDGE=as.numeric(coef(ridge)), 
                     LASSO=as.numeric(coef(lasso)), ENET=as.numeric(coef(enet)))


#################################################################
# Testing on verification set
#################################################################

# Function to print performance statistics
oos_fun <- function(outcome, predictor, name) {
  print(name)
  print(sprintf("Mean squared error: %.2f", mse_fun(outcome, predictor)))
  print(sprintf("RMSE: %.2f", sqrt(mse_fun(outcome, predictor))))
  print(sprintf("Coefficient of determination: %.2f", 
                r2_score_fun(outcome, predictor)))
}


# Print verification stats
cat("\n")
oos_fun(Y_train, predict(regr, as.matrix(X_tr_sd)), "OLS_tr")
oos_fun(Y_train, predict(ridge, as.matrix(X_tr_sd)), "RIDGE_tr")
oos_fun(Y_train, predict(lasso, as.matrix(X_tr_sd)), "LASSO_tr")
oos_fun(Y_train, predict(enet, as.matrix(X_tr_sd)), "ENET_tr")
cat("\n")  



# Print verification stats
cat("\n")
oos_fun(Y_verify, predict(regr, as.matrix(X_ve_sd)), "OLS_ve")
oos_fun(Y_verify, predict(ridge, as.matrix(X_ve_sd)), "RIDGE_ve")
oos_fun(Y_verify, predict(lasso, as.matrix(X_ve_sd)), "LASSO_ve")
oos_fun(Y_verify, predict(enet, as.matrix(X_ve_sd)), "ENET_ve")
cat("\n")  

