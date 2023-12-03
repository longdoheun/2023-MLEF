#install.packages("R6")
library(R6)
library(HDeconometrics)
library(randomForest)
library(xgboost)
library(sandwich)
library(MCS)

# for lstm
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
source("func-LSTM.R")
source("func-BILSTM.R")

Forecasting <- R6Class("Forecasting", list(
  colnames = NULL,
  # rolling window
  rw = NULL,
  ar = NULL,
  ridge = NULL,
  lasso = NULL,
  adalasso = NULL,
  elasticnet = NULL,
  adaelasticnet = NULL,
  rf = NULL,
  xg = NULL,
  lstm = NULL,
  bilstm = NULL,
  boruta.lstm = NULL,
  boruta.bilstm = NULL,
  lstm_sel = NULL,
  bilstm_sel = NULL,
  # inputs
  Y = NULL,
  real = NULL,
  window_size = NA,
  sample_size = NA,
  indice = NA,
  lag = NA,
  npred = NA,
  
  ### initiallization
  initialize = function(Y, window_size, indice, lag){
    self$Y = Y
    self$indice = indice
    self$window_size = window_size
    self$sample_size = nrow(Y)
    self$npred = nrow(Y) - window_size
    self$lag = lag
    self$real = tail(self$Y[,1], self$npred)
    
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
    
    # rolling window
    self$rw = self$rw.rolling()
    self$ar=self$ar.rolling()
    self$ridge=self$ridge.rolling()
    self$lasso=self$lasso.rolling()
    self$adalasso=self$adalasso.rolling()
    self$elasticnet=self$elasticnet.rolling()
    self$adaelasticnet=self$adaelasticnet.rolling()
    self$rf=self$rf.rolling()
    self$xg=self$xg.rolling()
    self$lstm = self$lstm.rolling()
    self$bilstm = self$bilstm.rolling()

    # boruta-cross validation
    self$boruta.lstm = self$boruta.lstm.selected()
    self$boruta.bilstm = self$boruta.bilstm.selected()

    # rolling window
    self$lstm_sel = self$lstm.sel.rolling()
    self$bilstm_sel = self$bilstm.sel.rolling()
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
  
  ## Boruta lstm cross-validation
  boruta.lstm.selected = function(){
    # cross validation with Boruta algorithm
    selected = cross_validation(
      self$Y,
      self$indice,
      self$lag,
      run_selected_lstm
    )
    
    return(selected)
  },
  
  ## Boruta bilstm cross-validation
  boruta.bilstm.selected = function(){
    # cross validation with Boruta algorithm
    selected = cross_validation(
      self$Y,
      self$indice,
      self$lag,
      run_selected_bilstm
    )
    
    return(selected)
  },
  
  ### 12. Boruta selected LSTM
  lstm.sel.rolling = function(){
    cat(glue::glue("LSTM selected rolling window : {self$lag}-step ahead \n"))
    
    # cross validation with Boruta algorithm
    selected = self$boruta.lstm
    
    lstm_sel <- selected.lstm.rolling.window(
      self$Y,
      self$npred,
      self$indice,
      self$lag,
      selected = selected
    )
    
    return(lstm_sel)
  },
  
  ### 13. Boruta selected BiLSTM
  bilstm.sel.rolling = function(){
    cat(glue::glue("BiLSTM selected rolling window : {self$lag}-step ahead \n"))
    
    # cross validation with Boruta algorithm
    selected = self$boruta.bilstm
    
    bilstm_sel <- selected.bilstm.rolling.window(
      self$Y,
      self$npred,
      self$indice,
      self$lag,
      selected = selected
    )
    
    return(bilstm_sel)
  },
  
  # pred
  get_prediction = function(){
    pred = cbind(
      self$rw$pred,
      self$ar$pred,
      self$ridge$pred,
      self$lasso$pred,
      self$adalasso$pred,
      self$elasticnet$pred,
      self$adaelasticnet$pred,
      self$rf$pred,
      self$xg$pred,
      self$lstm$pred,
      self$bilstm$pred,
      self$lstm_sel$pred,
      self$bilstm_sel$pred
    )
    
    colnames(pred) = self$colnames
    
    return(pred)
  },
  
  # RMSE
  get_RMSE = function(){
    RMSE = cbind(
      self$rw$error[1],
      self$ar$error[1],
      self$ridge$error[1],
      self$lasso$error[1],
      self$adalasso$error[1],
      self$elasticnet$error[1],
      self$adaelasticnet$error[1],
      self$rf$error[1],
      self$xg$error[1],
      self$lstm$error[1],
      self$bilstm$error[1],
      self$lstm_sel$error[1],
      self$bilstm_sel$error[1]
    )
    
    colnames(RMSE) = self$colnames
    
    return(RMSE)
  },
  
  # MAE
  get_MAE = function(){
    MAE = cbind(
      self$rw$error[2],
      self$ar$error[2],
      self$ridge$error[2],
      self$lasso$error[2],
      self$adalasso$error[2],
      self$elasticnet$error[2],
      self$adaelasticnet$error[2],
      self$rf$error[2],
      self$xg$error[2],
      self$lstm$error[2],
      self$bilstm$error[2],
      self$lstm_sel$error[2],
      self$bilstm_sel$error[2]
    )
    
    colnames(MAE) = self$colnames
    
    return(MAE)
  },
  
  # MCS test
  MCS_test = function(){
    pred = self$get_prediction()
    LOSS = (pred - self$real)^2 # squared error
    SSM <- MCSprocedure(LOSS, alpha=0.5, B=5000, statistic="Tmax")
    return(SSM)
  },
  
  # GW test
  GW_test = function(base_error){
    pred = self$get_prediction()
    test = matrix(NA, ncol(pred), 2)
    colnames(test) <- c("test-stat", "p-value")
    
    min_model_idx = which.min(base_error)
    cat("min_model=",self$colnames[min_model_idx])
    
    for (model_idx in 1:ncol(pred)) {
      if (model_idx == min_model_idx){
        test[model_idx,1] <- NA
        test[model_idx,2] <- NA
      } else{
        gw = gw.test(
          pred[,min_model_idx],
          pred[,model_idx],
          self$real,
          tau=self$lag,
          T=self$npred, 
          method="NeweyWest"
          )
        test[model_idx,1] <- gw$statistic
        test[model_idx,2] <- gw$p.value
      }
    }
    
    return(list("result"=test, "min_model"=min_model_idx))
  }
))


#==================================================================
library(readxl)
# Load Data Set
getwd()
dir()

Y=read_excel("transformed_data.xlsx")

Y = as.matrix(Y[,-1])
step_6_ahead <- Forecasting$new(Y, 120, 1, 6)
step_12_ahead <- Forecasting$new(Y, 120, 1, 12)
step_6_ahead.pred = step_6_ahead$get_prediction()
step_6_ahead.MAE = step_6_ahead$get_MAE()
step_6_ahead.RMSE = step_6_ahead$get_RMSE()
step_6_ahead$MCS_test()
step_6_ahead$GW_test(MAE)