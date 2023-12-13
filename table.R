### load all variables and functions
load("result_forecasting.RData")

# =========================================================
### Variable Ranking Table
var_ranks_12 = colnames(step12.ordered) %>% as.matrix()
var_ranks_6 = colnames(step6.ordered) %>% as.matrix()

boruta_var_ranks = cbind(var_ranks_6, matrix(nrow=nrow(var_ranks_6), ncol = 1), var_ranks_12, matrix(nrow = nrow(var_ranks_12), ncol = 1))

t = read_excel("금융시계열.xlsx", sheet = "전체 변수설명")


# 변수 설명을 찾아서 변수 순위별로 기입
for (i in 1:nrow(var_ranks_6)) {
  finding_idx = which(t$Code == var_ranks_6[i])
  boruta_var_ranks[i,2] = t$Description[finding_idx]
}

for (i in 1:nrow(var_ranks_6)) {
  finding_idx = which(t$Code == var_ranks_12[i])
  boruta_var_ranks[i,4] = t$Description[finding_idx]
}

colnames(boruta_var_ranks) = c("6-month-ahead","6d", "12-momth-ahead", "12d")

### Export result table to xlsx
library(openxlsx)
file_name = "boruta_variable.xlsx"

wb = createWorkbook()

addWorksheet(wb, sheetName = "boruta")
writeData(wb, sheet = "boruta", x=boruta_var_ranks)

saveWorkbook(wb, file=file_name)


# =========================================================
### Forecasting Performance table
performance = cbind(
  round(t(step6.RMSE),4),
  round(t(step6.RMSE)/t(step6.RMSE)[1],2),
  round(t(step6.MAE),4),
  round(t(step6.MAE)/t(step6.MAE)[1],2),
  round(t(step12.RMSE),4),
  round(t(step12.RMSE)/t(step12.RMSE)[1],2),
  round(t(step12.MAE),4),
  round(t(step12.MAE)/t(step12.MAE)[1],2)
)


colnames(performance) = c(
  "RMSE",
  "%",
  "MAE",
  "%",
  "RMSE",
  "%",
  "MAE",
  "%"
)

file_name = "performance.xlsx"

wb = createWorkbook()

addWorksheet(wb, sheetName = "pf")
writeData(wb, sheet = "pf", x=performance, rowNames = TRUE)

saveWorkbook(wb, file=file_name)


# =========================================================
### Giacomini-White Test Result

GW_6 = cbind(
  round(step6.GW_lstm,4),
  round(step6.GW_bilstm,4),
  round(step6.GW_lstm_sel,4),
  round(step6.GW_bilstm_sel,4)
)

GW_12 = cbind(
  round(step12.GW_lstm,4),
  round(step12.GW_bilstm,4),
  round(step12.GW_lstm_sel,4),
  round(step12.GW_bilstm_sel,4)
)

file_name = "gw.xlsx"

wb = createWorkbook()

addWorksheet(wb, sheetName = "GW_6")
writeData(wb, sheet = "GW_6", x=GW_6, rowNames = TRUE)

addWorksheet(wb, sheetName = "GW_12")
writeData(wb, sheet = "GW_12", x=GW_12, rowNames = TRUE)

saveWorkbook(wb, file=file_name)



