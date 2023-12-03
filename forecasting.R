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

# functions in func directory
source("func/func-rw.R")
source("func/func-ar.R")
source("func/func-lasso.R")
source("func/func-rf.R")
source("func/func-xgb.R")
source("func/func-boruta.R")
source("func/func-BILSTM-1.R")
source("func/func-LSTM-1.R")
source("func/gwtest.R")

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
