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
