setwd("/Users/doheun/Documents/R/ML/project")
dir()

### Boruta
library(Boruta)
library(dplyr)

# boruta without lagged variables
run_Boruta <- function(Y, indice=1, lag=1){
  
  comp=princomp(scale(Y,scale=FALSE))
  Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  aux=embed(Y3,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y3)*lag))]
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+4)] <- X[,(i+1)]
    X2[,(4*i+3)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+2)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+1)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  boruta = Boruta(X2, y, maxRuns = 100)
  plot = plot(boruta)
  plot
  attstats = attStats(boruta)
  
  order = order(attstats$meanImp, decreasing = T)
  ?Boruta
  return(list("order" = order, "attstats" = attstats, "variables"=X2, "ind"=y))
}

### Cross-validation for selecting the optimal number of variables
cross_validation <- function(Y, indice, lag, model){
  comp=princomp(scale(Y,scale=FALSE))
  Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
  
  # run Boruta Algorithm
  result <- run_Boruta(Y,indice,lag)
  y = result$ind
  
  Errors = rep(NA,70)
  Pred = rep(NA,70)
  
  for (i in 2:70){
    order = result$order
    cat("# of variable : ",i)
    selected = order[1:i]
    pred <- model(Y, indice, lag, selected)$pred
    y <- denormalize(y, min(Y2[,indice]),max(Y2[,indice]))
    error = mean((pred-y)^2)
    Errors[i] <- error
    Pred[i] <- pred
  }
  # plot(c(1:68), Errors, xlab="# of Variables", ylab="Fitted Squared Error")
  
  ### Rolling Window with Selected Variables
  selected = order[1:which.min(Errors)]           # The Set of Optimal Number of Variables
  
  return(list("selected" = selected, "pred" = Pred))
}








