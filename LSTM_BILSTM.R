
setwd("/Users/doheun/Documents/R/ML/project")
dir()

### Using LSTM Function  

# you first download anaconda and create(envname) & activate new environment
# check the envname and conda python version

# ==================================================================
# Install packages
install.packages("tidyverse")
install.packages("reshape2")
install.packages("tensorflow")  #관리자 권한으로 실행
install.packages("keras")  #관리자 권한으로 실행

# load packages
library(tensorflow)
library(keras)
install_keras(method = "conda", envname = "upbitpy", 
              conda_python_version = "3.11.4")

library(tidyverse)
library(ggplot2)
library(reshape2)


# reticulate으로 파이썬 가상환경 사용
library(reticulate)
use_condaenv("upbitpy", required = TRUE)
print(py_config())

# ==== check =====
#tf$constant("hello")
#tf$version

# ==================================================================
# Load Data Set

getwd()
dir()

load("rawdata.rda")

Y=dados

# ==================================================================
# US inflation forecasting example explanation
nprev=132
indice = 1
horizon = 1
lag = horizon

source("func-boruta-1.R")

boruta <- run_Boruta(Y,1,1)

b = boruta$order
a = boruta$variables

selected = cross_validation(Y, 1, 1)

source("func-LSTM.R")
# FINAL CHECK COMPLETE
lstm <- mul.lstm.rolling.window(Y,nprev,1,1)  
lstm$errors
lstm$pred

source("func-BILSTM.R")
bilstm <- mul.bilstm.rolling.window(Y,nprev,1,1)
bilstm$errors 
bilstm$pred

save.image("results_LSTM.RData")

load("results_LSTM.RData")

# ============================================================================
# Graphing Test Sets

date_test = seq(as.Date("1990-01-01"),as.Date("2000-12-01"), "months")

real_value = Y[,indice] %>% tail(nprev)

pred_value1 = multi_lstm_rolling_ex$pred
pred_value2 = bilstm$pred

eval = data.frame(date_test, real_value, pred_value1, pred_value2) %>% 
  set_names(c("Date","Actual","LSTM_Predicted", "BILSTM_Predicted")) %>% 
  reshape2::melt(id="Date")

ggplot(data = eval, aes(x=Date, y=value, colour=variable, group=variable)) +
  geom_line(size=0.5) +
  xlab("") + ylab("") + labs(color = "") +
  scale_x_date(date_breaks ="2 year", date_labels = "%Y-%m") + 
  ggtitle("Multivariate LSTM : Actual vs Predicted") 
