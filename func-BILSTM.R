# ==================================================================
# Normalization
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

# Inverse Normalization 
denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

# ==================================================================
# Multivariate LSTM Model 

run_multi_bilstm=function(Y,indice,horizon){
  
  comp=princomp(scale(Y,scale=FALSE))
  Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  aux=embed(Y3,4+horizon)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y3)*horizon))]  
  
  if(horizon==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y3)*(horizon-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+4)] <- X[,(i+1)]
    X2[,(4*i+3)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+2)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+1)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X.out2 <- X.out %>% replace(!0, 0)
  
  for(i in 0:(ncol(Y3)-1)){
    X.out2[(4*i+4)] <- X.out[(i+1)]
    X.out2[(4*i+3)] <- X.out[(i+ncol(Y3)+1)]
    X.out2[(4*i+2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2[(4*i+1)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr = array(
    data = as.numeric(unlist(X2)),
    dim = c(nrow(X), 4, ncol(Y3)))
  ?array
  X.out.arr = array(
    data = as.numeric(unlist(X.out2)),
    dim = c(1, 4, ncol(Y3)))
  dim(X.arr)
  # =============================================================
  set.seed(42)         # 신경망의 Initial weight를 설정할 때, Dropout과 같은 방법을 사용할 때 등에서 영향 
  set_random_seed(42)  #tensorflow의 경우 tensorflow::set_random_seed()  함수를 사용해서 시드 설정
  
  
  
  # Hyper-Parameters Adjustment
  
  batch_size = 25  # 25 또는 32 한 번에 입력하는 데이터 크기 
  feature = ncol(Y3)  # 설명변수 수 
  epochs = 100  # 학습 횟수, 100
  
  model = keras_model_sequential()
  
  # 1-layer model 실행
  
  model %>%  bidirectional(layer_lstm(units = 32, input_shape = c(4, feature),stateful = FALSE)) %>% layer_dense(units = 1) 
  
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  model %>% summary()
  
  history = model %>% fit(X.arr, y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  # =============================================================
  
  pred = model(X.out.arr) %>% denormalize(min(Y2[,indice]),max(Y2[,indice])) # De-normalization
  
  
  return(list("model"=model,"pred"=pred))
}


# ============================================================================
# Multivariate Rolling n-Step ahead LSTM Forecast

mul.bilstm.rolling.window=function(Y,nprev,indice=1,lag=1){
  
  save.pred=matrix(NA,nprev,1)
  
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] %>% as.data.frame()
    bilstm=run_multi_bilstm(Y.window,indice,lag)
    save.pred[(1+nprev-i),]=as.numeric(bilstm$pred) # Note as.numeric()
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
  mae=mean(abs(tail(real,nprev)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

# ==================================================================
# variable selected Multivariate LSTM Model 

run_selected_bilstm <- function(Y, indice=1, horizon=1, selected){
  sel = c(1, unlist(selected)+1)
  Y2 = Y[,sel] %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  aux=embed(Y3,4+horizon)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y3)*horizon))]
  
  if(horizon==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y3)*(horizon-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+4)] <- X[,(i+1)]
    X2[,(4*i+3)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+2)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+1)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X.out2 <- X.out %>% replace(!0, 0)
  
  for(i in 0:(ncol(Y3)-1)){
    X.out2[(4*i+4)] <- X.out[(i+1)]
    X.out2[(4*i+3)] <- X.out[(i+ncol(Y3)+1)]
    X.out2[(4*i+2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2[(4*i+1)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr = array(
    data = as.numeric(unlist(X2)),
    dim = c(nrow(X), 4, ncol(Y3)))
  ?array
  X.out.arr = array(
    data = as.numeric(unlist(X.out2)),
    dim = c(1, 4, ncol(Y3)))
  dim(X.arr)
  # =============================================================
  set.seed(42)         # 신경망의 Initial weight를 설정할 때, Dropout과 같은 방법을 사용할 때 등에서 영향 
  set_random_seed(42)  #tensorflow의 경우 tensorflow::set_random_seed()  함수를 사용해서 시드 설정
  
  # Hyper-Parameters Adjustment
  batch_size = 25  # 25 또는 32 한 번에 입력하는 데이터 크기 
  feature = ncol(Y3)  # 설명변수 수 
  epochs = 100  # 학습 횟수, 100
  
  model = keras_model_sequential()
  
  # 1-layer model 실행
  model %>% bidirectional(layer_lstm(units = 32, input_shape = c(4, feature),stateful = FALSE)) %>% layer_dense(units = 1)
  
  model %>% compile(loss = 'mse', optimizer = 'adam')
  model %>% summary()
  history = model %>% fit(X.arr, y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  # =============================================================
  # De-normalization
  pred = model(X.out.arr) %>% denormalize(min(Y2[,indice]),max(Y2[,indice])) %>% as.numeric()
  return(list("model"=model,"pred"=pred))
}

# ============================================================================
# variable selected Multivariate Rolling n-Step ahead LSTM Forecast

selected.bilstm.rolling.window=function(Y,nprev,indice=1,lag=1, selected){
  
  save.pred=matrix(NA,nprev,1)
  
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] %>% as.data.frame()
    bilstm=run_selected_bilstm(Y.window,indice,lag, selected)
    save.pred[(1+nprev-i),]=as.numeric(bilstm$pred) # Note as.numeric()
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
  mae=mean(abs(tail(real,nprev)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}