### Boruta
library(Boruta)
library(dplyr)

# boruta without lagged variables
run_Boruta <- function(Y, indice=1, lag=1){
  y=Y[,1]
  X=Y[,-1]
  
  boruta = Boruta(X, y, maxRuns = 100)
  plot = plot(boruta)
  plot
  attstats = attStats(boruta)
  
  order = order(attstats$meanImp, decreasing = T)
  ?Boruta
  return(list("order" = order, "attstats" = attstats, "variables"=X, "ind"=y))
}

### Cross-validation for selecting the optimal number of variables
cross_validation <- function(Y, indice, lag, model){
  
  # run Boruta Algorithm
  result <- run_Boruta(Y,indice,lag)
  X = result$variables
  y = result$ind
  
  Errors = rep(NA,ncol(X))
  
  for (i in 2:ncol(X)){
    order = result$order
    cat("# of variable : ",i)
    selected = order[1:i]
    pred <- model(Y, indice, lag, selected)$pred
    error = mean((pred-y)^2)
    Errors[i] <- error
  }
  # plot(c(1:68), Errors, xlab="# of Variables", ylab="Fitted Squared Error")
  
  ### Rolling Window with Selected Variables
  selected = order[1:which.min(Errors)]           # The Set of Optimal Number of Variables
  
  return(list("selected" = selected))
}