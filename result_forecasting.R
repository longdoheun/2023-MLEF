setwd("/Users/doheun/Documents/R/ML/MLEF_project")


### this is for LSTM
library(reticulate) # reticulate:: using python env
use_condaenv("upbitpy", required = TRUE) # In my case, conda env name is "upbitpy"
print(py_config())


source("Forecasting.R")
#==================================================================
# Load Data Set
library(readxl)

### load all variables and functions
load("result_forecasting.RData")

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



### load the transformed independent variables
X = read_excel("transformed_data.xlsx")
# ======== you can check the tcode transformation in "Tcode_transformation.Rmd" file.=============

date = X[,1]
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
## tgt_model_idx : alternative model index
tgt_model_idx = which.min(step6.MAE) # MAE 기준
## Check all alternative models (10,11,12,13) with benchmarks as Kim and Han (2022)
step6.GW_lstm = step6$GW_test(step6.pred,10) # LSTM = 10
step6.GW_bilstm = step6$GW_test(step6.pred, 11) # BiLSTM = 11
step6.GW_lstm_sel = step6$GW_test(step6.pred,12) # LSTM_sel = 12
step6.GW_bilstm_sel = step6$GW_test(step6.pred,13) # BiLSTM_sel = 13

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
## tgt_model_idx : alternative model index
tgt_model_idx = which.min(step12.MAE) # MAE 기준일 경우

## Check all alternative models (10,11,12,13) with benchmarks as Kim and Han (2022)
step12.GW_lstm = step12$GW_test(step12.pred,10) # LSTM = 10
step12.GW_bilstm = step12$GW_test(step12.pred, 11) # BiLSTM = 11
step12.GW_lstm_sel = step12$GW_test(step12.pred,12) # LSTM_sel = 12
step12.GW_bilstm_sel = step12$GW_test(step12.pred,13) # BiLSTM_sel = 13

# Out-of-sample forecasting result for conclusion
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


### save all the current variables
save.image("result_forecasting.RData")