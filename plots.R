### load all variables and functions
load("result_forecasting.RData")

source("Forecasting.R")
# ============================================================================
# Graphing Test Sets

date_test = seq(as.Date("2007-01-01"),as.Date("2023-06-01"), "months")

real_value = APTS_S_12

eval = data.frame(date_test, real_value) %>%
  set_names(c("Date","Year-over-Year Growth Rate")) %>% 
  reshape2::melt(id="Date")

ggplot(data = eval, aes(x=Date, y=value, colour=variable, group=variable)) +
  geom_line(size=0.5) +
  xlab("") + ylab("") + labs(color = "") +
  scale_x_date(date_breaks ="2 year", date_labels = "%Y-%m") + 
  ggtitle("Seoul Apartment Transaction Price Index")

# ============================================================================
library(readxl)
raw_data = read_excel("raw_data.xlsx")
tdata = raw_data[-1,1]

tdata = tdata %>% mutate(date = date %>% as.integer(),
                 date = as.Date("1899-12-30") + date)


eval = data.frame(tdata, y[,1]) %>% 
  set_names(c("Date","22-Jan = 100")) %>% 
  reshape2::melt(id="Date")

ggplot(data = eval, aes(x=Date, y=value, colour=variable, group=variable)) +
  geom_line(size=0.5) +
  xlab("") + ylab("") + labs(color = "") +
  scale_x_date(date_breaks ="2 year", date_labels = "%Y-%m") + 
  ggtitle("Seoul Apartment Transaction Price Index")

#=============================================================================
library(ggplot2)
library(data.table)
install.packages("data.table")

start_date <- as.Date("2017-01-01")
end_date <- as.Date("2023-06-01")

date_list <- seq(start_date, end_date, by = "months") %>% as.data.table() 
colnames(date_list) <- "date"

apt <- Y[121:198,] %>% as.data.table() %>% dplyr::select(Price_APT_S)

apt2 <- apt %>% cbind(step6.pred$lstm_sel) %>% cbind(step12.pred$bilstm_sel) %>% cbind(date_list)
colnames(apt2) <- c("APT", "APT6", "APT12", "date")

# 만약 시간 변수가 데이터프레임에 존재한다면, 예를 들어 "Time"이라는 열이 있다면:
ggplot(apt2, aes(x = date)) +
  geom_line(aes(y = APT6, color = "APT6"), size = 0.5, linetype = "dashed") +  # APT6에 대한 선 그래프 추가
  geom_line(aes(y = APT12, color = "APT12"), size = 0.5, linetype = "dashed") +  # APT12에 대한 선 그래프 추가
  geom_line(aes(y = APT, color = "APT"), size = 1) +  # APT12에 대한 선 그래프 추가
  labs(x = "", y = "APT_price") +
  scale_color_manual(values = c(APT6 = "red", APT12 = "blue", APT = "black"), 
                     labels = c("Real Value", "LSTM_sel_6m", "Bi LSTM_sel_12m")) +
  theme_minimal() +
  theme(axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 12))
