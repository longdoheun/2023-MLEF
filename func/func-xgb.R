runxgb=function(Y,indice,horizon){
  
  comp=princomp(scale(Y,scale=FALSE))
  Y2=cbind(Y,comp$scores[,1:4])
  aux=embed(Y2,4+horizon)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*horizon))]  
  
  if(horizon==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(horizon-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model = xgboost(X,label = y,nrounds = 1000, verbose = FALSE,
                  params=list(eta=0.05,nthread=1,colsample_bylevel=2/3,subsample=1,max_depth=4,min_child_weight=nrow(X)/200))
  
  pred=predict(model,t(X.out))
  
  return(list("model"=model,"pred"=pred))
}


xgb.rolling.window=function(Y,npred,indice=1,horizon=1){
  
  save.pred=matrix(NA,npred,1)
  for(i in npred:1){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runxgb(Y.window,indice,horizon)
    save.pred[(1+npred-i),]=lasso$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred)-save.pred)^2))
  mae=mean(abs(tail(real,npred)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

