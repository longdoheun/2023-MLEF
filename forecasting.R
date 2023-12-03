setwd("/Users/doheun/Documents/R/ML/project")

#install.packages("R6")
library(R6)
library(HDeconometrics)
library(randomForest)
library(xgboost)
library(sandwich)
library(MCS)

# packages for lstm
library(tensorflow)
library(keras)
library(tidyverse)
library(ggplot2)
library(reshape2)

library(reticulate) # reticulate:: using python env
use_condaenv("upbitpy", required = TRUE) # In my case, conda env name is "upbitpy"
print(py_config())

# functions
source("func-rw.R")
source("func-ar.R")
source("func-lasso.R")
source("func-rf.R")
source("func-xgb.R")
source("gwtest.R")
source("func-boruta.R")
source("func-LSTM-1.R")
source("func-BILSTM-1.R")

Forecasting <- R6Class("Forecasting", list(
  colnames = NULL,

  # inputs
  Y = NULL,
  real = NULL,
  window_size = NA,
  sample_size = NA,
  indice = NA,
  lag = NA,
  npred = NA,
  borutaResult = NULL,
  
  ### initialize the R6 class
  initialize = function(Y, window_size, indice, lag){
    self$Y = Y
    self$indice = indice
    self$window_size = window_size
    self$sample_size = nrow(Y)
    self$npred = nrow(Y) - window_size
    self$lag = lag
    self$real = tail(self$Y[,1], self$npred)
    self$borutaResult = self$boruta()
    
    self$colnames = c(
      "rw",
      "ar",
      "ridge",
      "lasso",
      "adalasso",
      "elasticnet",
      "adaelasticnet",
      "random forest",
      "XGboost",
      "lstm",
      "bilstm",
      "lstm_sel",
      "bilstm_sel"
    )
    
    #====================================================
    ### not recommended
    ### Execute the rolling window when initialzing the class 
    # self$rw = self$rw.rolling()
    # self$ar=self$ar.rolling()
    # self$ridge=self$ridge.rolling()
    # self$lasso=self$lasso.rolling()
    # self$adalasso=self$adalasso.rolling()
    # self$elasticnet=self$elasticnet.rolling()
    # self$adaelasticnet=self$adaelasticnet.rolling()
    # self$rf=self$rf.rolling()
    # self$xg=self$xg.rolling()
    # self$lstm = self$lstm.rolling()
    # self$bilstm = self$bilstm.rolling()
    # 
    # # boruta-cross validation
    # self$boruta.lstm = self$boruta.lstm.selected()
    # self$boruta.bilstm = self$boruta.bilstm.selected()
    # 
    # # rolling window
    # self$lstm_sel = self$lstm.sel.rolling()
    # self$bilstm_sel = self$bilstm.sel.rolling()
    #======================================================
  },
  
  ### 1.Random Walk
  rw.rolling = function(){
    cat(glue::glue("RW rolling window : {self$lag}-step ahead \n"))
    
    rw <- rw.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag)
    
    return(rw)
  },
  
  ### 2. AR(4)
  ar.rolling = function(){
    cat(glue::glue("AR(4) rolling window : {self$lag}-step ahead \n"))
    
    ar <- ar.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag)
    
    return(ar)
  },
  
  ### 3. Ridge regression
  ridge.rolling = function(){
    cat(glue::glue("Ridge Regression rolling window : {self$lag}-step ahead \n"))
    
    alpha = 0
    
    ridge <- lasso.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag,
      alpha,
      type="lasso")
    
    return(ridge)
  },
  
  ### 4. LASSO
  lasso.rolling = function(){
    cat(glue::glue("LASSO rolling window : {self$lag}-step ahead \n"))
    alpha = 1
    
    lasso <- lasso.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag,
      alpha,
      type="lasso")
    
    return(lasso)
  },
  
  ### 5. Adaptive LASSO
  adalasso.rolling = function(){
    cat(glue::glue("adaptive LASSO rolling window : {self$lag}-step ahead \n"))
    
    alpha = 1
    
    adalasso <- lasso.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag,
      alpha,
      type="adalasso")
    
    return(adalasso)
  },
  
  ### 6. Elastic Net
  elasticnet.rolling = function(){
    cat(glue::glue("Elastic Net rolling window : {self$lag}-step ahead \n"))
    
    alpha = 0.5
    
    elasticnet <- lasso.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag,
      alpha,
      type="lasso")
    
    return(elasticnet)
  },
  
  ### 7. Adaptive Elastic Net
  adaelasticnet.rolling = function(){
    cat(glue::glue("Adaptive Elastic Net rolling window : {self$lag}-step ahead \n"))
    
    alpha = 0.5
    
    adaelasticnet <- lasso.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag,
      alpha,
      type="adalasso")
    
    return(adaelasticnet)
  },
  
  ### 8. Random Forest
  rf.rolling = function(){
    cat(glue::glue("random forest rolling window : {self$lag}-step ahead \n"))
    
    rf <- rf.rolling.window(
      self$Y, 
      self$npred,
      self$indice,
      self$lag)
    
    return(rf)
  },
  
  ### 9. XGboost
  xg.rolling = function(){
    cat(glue::glue("xgboost rolling window : {self$lag}-step ahead \n"))
    
    xg <- xgb.rolling.window(
      self$Y,
      self$npred,
      self$indice,
      self$lag
    )
    
    return(xg)
  },
  
  ### 10. LSTM
  lstm.rolling = function(){
    cat(glue::glue("LSTM rolling window : {self$lag}-step ahead \n"))
    
    lstm <- mul.lstm.rolling.window(
      self$Y,
      self$npred,
      self$indice,
      self$lag
    )
    
    return(lstm)
  },
  
  ### 11. BiLSTM
  bilstm.rolling = function(){
    cat(glue::glue("BiLSTM rolling window : {self$lag}-step ahead \n"))
    
    bilstm <- mul.bilstm.rolling.window(
      self$Y,
      self$npred,
      self$indice,
      self$lag
    )
    
    return(bilstm)
  },
  
  ## cross validation for Boruta-lstm # CV was not used
  boruta.lstm.selected = function(){
    selected = cross_validation(
      self$Y,
      self$indice,
      self$lag,
      run_selected_lstm
    )
    
    return(selected)
  },
  
  ## cross validation for Boruta-BiLSTM # CV was not used
  boruta.bilstm.selected = function(){
    selected = cross_validation(
      self$Y,
      self$indice,
      self$lag,
      run_selected_bilstm
    )
    
    return(selected)
  },
  
  ## Boruta algorithm
  boruta = function(){
    B = run_Boruta(
      self$Y,
      self$indice,
      self$lag)
    
    X = self$Y[,-1]
    X_re <- X[,unlist(B$order)]
    
    confirmed = X[,colnames(X)[B$attstats$decision == "Confirmed"]]
    
    return(list("ordered"=X_re, "confirmed"=confirmed))
  },
  
  ### 12. Boruta selected LSTM
  lstm.sel.rolling = function(){
    cat(glue::glue("LSTM selected rolling window : {self$lag}-step ahead \n"))
    
    Y = cbind(self$Y[,1], self$borutaResult$confirmed) # combine dependent variable with confirmed covariates
    
    lstm_sel <- mul.lstm.rolling.window(
      Y,
      self$npred,
      self$indice,
      self$lag
    )
    
    return(lstm_sel)
  },
  
  ### 13. Boruta selected BiLSTM
  bilstm.sel.rolling = function(){
    cat(glue::glue("BiLSTM selected rolling window : {self$lag}-step ahead \n"))
    
    Y = cbind(self$Y[,1], self$borutaResult$confirmed) # combine independent variable with confirmed covariates
    
    bilstm_sel <- mul.bilstm.rolling.window(
      Y,
      self$npred,
      self$indice,
      self$lag
    )
    
    return(bilstm_sel)
  },
  
  # pred
  get_prediction = function(...){
    models <- list(...)
    pred <- do.call(data.frame, models %>% lapply(function(model) model$pred) )
    colnames(pred) = self$colnames[1:length(models)]
    
    return(pred)
  },
  
  # RMSE
  get_RMSE = function(...){
    models <- list(...)
    RMSE <- do.call(data.frame, models %>% lapply(function(model) model$error[1]) )
    
    colnames(RMSE) = self$colnames[1:length(models)]
    
    return(RMSE)
  },
  
  # MAE
  get_MAE = function(...){
    models <- list(...)
    MAE <- do.call(data.frame, models %>% lapply(function(model) model$error[2]) )
    
    colnames(MAE) = self$colnames[1:length(models)]
    
    return(MAE)
  },
  
  # MCS test
  MCS_test = function(pred){
    LOSS = (pred - self$real)^2 # squared error
    SSM <- MCSprocedure(LOSS, alpha=0.5, B=5000, statistic="Tmax")
    return(SSM)
  },
  
  # GW test
  GW_test = function(pred, tgt_model_idx){
    test = matrix(NA, ncol(pred), 2)
    colnames(test) <- c("test-stat", "p-value")

    cat("tgt_model=",self$colnames[tgt_model_idx])
    
    for (model_idx in 1:ncol(pred)) {
      if (model_idx == tgt_model_idx){ # if it is tgt model, fill NA
        test[model_idx,1] <- NA
        test[model_idx,2] <- NA
      } else{
        gw = gw.test(
          pred[,model_idx],
          pred[,tgt_model_idx],
          self$real,
          tau=self$lag,
          T=self$npred, 
          method="NeweyWest"
          )
        test[model_idx,1] <- gw$statistic
        test[model_idx,2] <- gw$p.value
      }
    }
    
    rownames(test) = step6$colnames
    
    return(test)
  }
))


#==================================================================
library(readxl)
# Load Data Set
getwd()
dir()

### load all variables and functions
#load("result_forecasting.RData")

### load lstm/Bilstm result 
# load("step6_lstm.Rdata")
# load("step6_bilstm.Rdata")
# load("step12_lstm.Rdata")
# load("step12_bilstm.Rdata")

### load boruta lstm/Bilstm result
# load("step6_lstm_sel.Rdata")
# load("step6_bilstm_sel.Rdata")
# load("step12_lstm_sel.Rdata")
# load("step12_bilstm_sel.Rdata")

#==================================================================
## 6-step ahead forecasting
# # of models : 13

y = read_excel("raw_data.xlsx") # raw data which is not transformed
y = as.matrix(y[-1,-1]) # drop the date column(1) and tcode row(1)

APTS_S_12 = as.matrix(diff(log(y[,1]), 12)) # log 1-year differences; dependent variable
colnames(APTS_S_12) = "Price_APT_S"

X = read_excel("transformed_data.xlsx") # load the transformed independent variables
X = as.matrix(X[,-1]) # drop the date column

Y = cbind(APTS_S_12, X)

step6 <- Forecasting$new(Y, 120, 1, 6)

# variable selection using Boruta Algorithm
step6.ordered = step6$borutaResult$ordered # ordered variables
step6.confirmed = step6$borutaResult$confirmed # confirmed variables use in LSTM/BiLSTM

step6.rw = step6$rw.rolling()
step6.ar = step6$ar.rolling()
step6.ridge = step6$ridge.rolling()
step6.lasso = step6$lasso.rolling()
step6.adalasso = step6$adalasso.rolling()
step6.elasticnet = step6$elasticnet.rolling()
step6.adaelasticnet = step6$adaelasticnet.rolling()
step6.rf = step6$rf.rolling()
step6.xg = step6$xg.rolling()

step6.lstm = step6$lstm.rolling()
# save(step6.lstm, file = "step6_lstm.Rdata")
step6.bilstm = step6$bilstm.rolling()
# save(step6.bilstm, file = "step6_bilstm.Rdata")
step6.lstm_sel = step6$lstm.sel.rolling()
# # save(step6.lstm_sel, file = "step6_lstm_sel.Rdata")
step6.bilstm_sel = step6$bilstm.sel.rolling()
# save(step6.bilstm_sel, file = "step6_bilstm_sel.Rdata")

step6.pred = step6$get_prediction(
  step6.rw,
  step6.ar,
  step6.ridge,
  step6.lasso,
  step6.adalasso,
  step6.elasticnet,
  step6.adaelasticnet,
  step6.rf,
  step6.xg,
  step6.lstm,
  step6.bilstm,
  step6.lstm_sel,
  step6.bilstm_sel
)

step6.MAE = step6$get_MAE(
  step6.rw,
  step6.ar,
  step6.ridge,
  step6.lasso,
  step6.adalasso,
  step6.elasticnet,
  step6.adaelasticnet,
  step6.rf,
  step6.xg,
  step6.lstm,
  step6.bilstm,
  step6.lstm_sel,
  step6.bilstm_sel
)

step6.RMSE = step6$get_RMSE(
  step6.rw,
  step6.ar,
  step6.ridge,
  step6.lasso,
  step6.adalasso,
  step6.elasticnet,
  step6.adaelasticnet,
  step6.rf,
  step6.xg,
  step6.lstm,
  step6.bilstm,
  step6.lstm_sel,
  step6.bilstm_sel
)

# MCS test
step6$MCS_test(step6.pred)

## Giacomini - White test
## tgt_model_idx : base model index
tgt_model_idx = which.min(step6.MAE) # MAE 기준
## Check all alternative models (10,11,12,13) with benchmarks as Kim and Han (2022)
step6.GW_lstm = step6$GW_test(step6.pred,10) # LSTM = 10
step6.GW_bilstm = step6$GW_test(step6.pred, 11) # BiLSTM = 11
step6.GW_lstm_sel = step6$GW_test(step6.pred,12) # LSTM_sel = 12
step6.GW_bilstm_sel = step6$GW_test(step6.pred,13) # BiLSTM = 13

#=========================================================
## 12-step ahead forecasting
# # of models : 13

step12 <- Forecasting$new(Y, 120, 1, 12)

# variable selection using Boruta Algoritnm
step12.ordered = step12$borutaResult$ordered # ordered variables
step12.confirmed = step12$borutaResult$confirmed # confirmed variables use in LSTM/BiLSTM

step12.rw = step12$rw.rolling()
step12.ar = step12$ar.rolling()
step12.ridge = step12$ridge.rolling()
step12.lasso = step12$lasso.rolling()
step12.adalasso = step12$adalasso.rolling()
step12.elasticnet = step12$elasticnet.rolling()
step12.adaelasticnet = step12$adaelasticnet.rolling()
step12.rf = step12$rf.rolling()
step12.xg = step12$xg.rolling()

step12.lstm = step12$lstm.rolling()
# save(step12.lstm, file = "step12_lstm.Rdata")
step12.bilstm = step12$bilstm.rolling()
# save(step12.bilstm, file = "step12_bilstm.Rdata")
step12.lstm_sel = step12$lstm.sel.rolling()
# # save(step12.lstm_sel, file = "step12_lstm_sel.Rdata")
step12.bilstm_sel = step12$bilstm.sel.rolling()
# # save(step12.bilstm_sel, file = "step12_bilstm_sel.Rdata")

step12.pred = step12$get_prediction(
  step12.rw,
  step12.ar,
  step12.ridge,
  step12.lasso,
  step12.adalasso,
  step12.elasticnet,
  step12.adaelasticnet,
  step12.rf,
  step12.xg,
  step12.lstm,
  step12.bilstm,
  step12.lstm_sel,
  step12.bilstm_sel
  )
step12.MAE = step12$get_MAE(
  step12.rw,
  step12.ar,
  step12.ridge,
  step12.lasso,
  step12.adalasso,
  step12.elasticnet,
  step12.adaelasticnet,
  step12.rf,
  step12.xg,
  step12.lstm,
  step12.bilstm,
  step12.lstm_sel,
  step12.bilstm_sel
  )
step12.RMSE = step12$get_RMSE(
  step12.rw,
  step12.ar,
  step12.ridge,
  step12.lasso,
  step12.adalasso,
  step12.elasticnet,
  step12.adaelasticnet,
  step12.rf,
  step12.xg,
  step12.lstm,
  step12.bilstm,
  step12.lstm_sel,
  step12.bilstm_sel
  )


# MCS test
step12$MCS_test(step12.pred)

## Giacomini - White test
## tgt_model_idx : base model index
tgt_model_idx = which.min(step12.MAE) # MAE 기준일 경우

## Check all alternative models (10,11,12,13) with benchmarks as Kim and Han (2022)
step12.GW_lstm = step12$GW_test(step12.pred,10) # LSTM = 10
step12.GW_bilstm = step12$GW_test(step12.pred, 11) # BiLSTM = 11
step12.GW_lstm_sel = step12$GW_test(step12.pred,12) # LSTM_sel = 12
step12.GW_bilstm_sel = step12$GW_test(step12.pred,13) # BiLSTM = 13

# Out-of-sample forecasting
OOS_forecasting = matrix(NA,4,2)
colnames(OOS_forecasting) <- c("6 month", "12 month")
rownames(OOS_forecasting) <- c("LSTM", "BiLSTM", "LSTM_sel", "BiLSTM_sel")

for (i in 1:2){
  if (i==1) {
    Y_ = cbind(Y[,1], step6.confirmed)
  } else {
    Y_ = cbind(Y[,1], step12.confirmed)
  }
  OOS_forecasting[1,i] = as.numeric(run_multi_lstm(Y,1,i*6)$pred)
  OOS_forecasting[2,i] = as.numeric(run_multi_bilstm(Y,1,i*6)$pred)
  OOS_forecasting[3,i] = as.numeric(run_multi_lstm(Y_,1,i*6)$pred)
  OOS_forecasting[4,i] = as.numeric(run_multi_bilstm(Y_,1,i*6)$pred)
}


### current save all the variables
save.image("result_forecasting.RData")
