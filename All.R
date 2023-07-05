
library(data.table)
library(lubridate)
library(tidyr)
library(tidyverse)
library(forecast)
library(outliers)
library(doParallel)
options(scipen = 999)
library(forecast)
save_metrics <- function(prediccion, grown_truth) {
  RMSE <- sqrt(mean((grown_truth - prediccion)^2))
  ERROR <- abs(sum(grown_truth) - sum(prediccion)) / (sum(grown_truth))
  out <- data.frame(RMSE = RMSE, ERROR = ERROR)
  return(out)
}
estandarizar <- function(x, mu, sigma, reverse = F) {
  if (missing(mu))
    mu <- mean(x)
  if (missing(sigma))
    sigma <- sd(x)
  if (reverse) {
    z <- x * sigma + mu
  } else {
    z <- (x - mu) / sigma
  }
  if (all(x == 0)) {
    z <- rep(0, length(x))
  }
  out <- list(z = z, mu = mu, sigma = sigma)
  return(out)
}

ventas <- fread("datos.csv", sep=",",
                data.table = F)
festivos <- read.csv("festivos.csv", sep = ";") %>% 
  mutate(FECHA = as.Date(FECHA)) %>% 
  filter(FECHA >= as.Date("2021-10-22") & FECHA <= as.Date("2022-11-10"))
t <- 1:385
dias <- day(festivos$FECHA)
x.reg <- festivos$FESTIVO
load("Pretuneado.RData")
top_modelos <- results %>% group_by(PDV, SKU) %>% 
  slice_min(RMSE, n = 1, with_ties = F) %>% ungroup()

skus <- unique(ventas$SKU)

{cl <- makeCluster(min(length(skus), detectCores() - 2))
  registerDoParallel(cl)} # Paralelización

start <- Sys.time() # Verbosity
results_list <- foreach (sku = skus, .packages = c("dplyr","lubridate","forecast","data.table","keras","tidyr")) %dopar% {
  
  ventas.sku <- ventas[ventas$SKU == sku,]
  top_modelos.sku <- top_modelos[top_modelos$SKU == sku,]
  pdvs <- unique(ventas.sku$PDV)
  results <- list() # Inicializamos output
  
  rm(ventas) # Ahorro memoria
  
  # Redes Neuronales ----
  
  apiladas <- ventas.sku %>% pivot_wider(names_from = FECHA, values_from = VENTA)
  info <- apiladas[,1:3]
  matriz <- as.matrix(apiladas[,4:388]); colnames(matriz) <- NULL
  estand_filas <- apply(matriz[,1:375], 1, estandarizar)
  input <- do.call(rbind, sapply(estand_filas, "[", 1))
  
  ## FNN ----
  t1 <- Sys.time()
  fnn <- keras_model_sequential()
  fnn %>%
    layer_dense(units = 3, activation = "tanh", input_shape = c(365)) %>% 
    layer_dense(units = 10, activation = "linear")
  fnn %>% compile(loss = "mse", optimizer = "SGD")
  fnn %>% fit(input[,1:365], input[,366:375], epochs = 100, batch_size = nrow(input), verbose = 0)
  t2 <- Sys.time()
  t3 <- Sys.time()
  pred_fnn <- fnn %>% predict(input[,11:375], verbose = 0)
  t4 <- Sys.time()
  train.cost_fnn <- difftime(t2, t1, units = "secs")
  fit.cost_fnn <- difftime(t4, t3, units = "secs")
  
  ## RNN ----
  t1 <- Sys.time()
  rnn <- keras_model_sequential()
  rnn %>%
    layer_simple_rnn(units = 1, activation = "tanh", input_shape = c(365,1)) %>% 
    layer_dense(units = 10, activation = "linear")
  rnn %>% compile(loss = "mse",
                     optimizer = optimizer_sgd(learning_rate = 0.1))
  rnn %>% fit(input[,1:365], input[,366:375], epochs = 3, batch_size = nrow(input), verbose = 0)
  t2 <- Sys.time()
  t3 <- Sys.time()
  pred_rnn <- rnn %>% predict(input[,11:375], verbose = 0)
  t4 <- Sys.time()
  train.cost_rnn <- difftime(t2, t1, units = "secs")
  fit.cost_rnn <- difftime(t4, t3, units = "secs")
  
  ## LSTM ----
  t1 <- Sys.time()
  lstm <- keras_model_sequential()
  lstm %>%
    layer_lstm(units = 1, activation = "tanh", input_shape = c(365,1)) %>% 
    layer_dense(units = 10, activation = "linear")
  lstm %>% compile(loss = "mse",
                  optimizer = optimizer_sgd(learning_rate = 0.01))
  lstm %>% fit(input[,1:365], input[,366:375], epochs = 5, batch_size = nrow(input), verbose = 0)
  t2 <- Sys.time()
  t3 <- Sys.time()
  pred_lstm <- lstm %>% predict(input[,11:375], verbose = 0)
  t4 <- Sys.time()
  train.cost_lstm <- difftime(t2, t1, units = "secs")
  fit.cost_lstm <- difftime(t4, t3, units = "secs")
  
  for (pdv in pdvs) {
    
    ventas.pdv <- ventas.sku[ventas.sku$PDV == pdv,]
    top_modelos.pdv <- top_modelos.sku[top_modelos.sku$PDV == pdv,]
    
    # Si no tiene datos de entrenmaiento, pasamos
    if (all(ventas.pdv$VENTA[1:365] == 0)) next
    
    abc <- unique(ventas.pdv$ABC) # Guardamos segmento
    
    # Redes
    # FNN
    results_aux <- save_metrics(prediccion = pred_fnn[info$PDV == pdv,],
                            grown_truth = ventas.pdv$VENTA[376:385]) %>% 
      mutate(PDV = pdv, SKU = sku, ABC = abc, MODELO = "FNN",
             TRAIN.COST = train.cost_fnn, FIT.COST = fit.cost_fnn)
    results[[length(results)+1]] <- results_aux
    # RNN
    results_aux <- save_metrics(prediccion = pred_rnn[info$PDV == pdv,],
                                grown_truth = ventas.pdv$VENTA[376:385]) %>% 
      mutate(PDV = pdv, SKU = sku, ABC = abc, MODELO = "RNN",
             TRAIN.COST = train.cost_rnn, FIT.COST = fit.cost_rnn)
    results[[length(results)+1]] <- results_aux
    # LSTM
    results_aux <- save_metrics(prediccion = pred_lstm[info$PDV == pdv,],
                                grown_truth = ventas.pdv$VENTA[376:385]) %>% 
      mutate(PDV = pdv, SKU = sku, ABC = abc, MODELO = "LSTM",
             TRAIN.COST = train.cost_lstm, FIT.COST = fit.cost_lstm)
    results[[length(results)+1]] <- results_aux
    
    # Determinísticos ----
    
    ## PreBDR ----
    
    df <- data.frame(Y = ventas.pdv$VENTA) %>% 
      cbind(seasonaldummy(ts(t, frequency = 7)))
    t1 <- Sys.time()
    modelo <- lm(Y ~ ., df[1:375,])
    t2 <- Sys.time()
    t3 <- Sys.time()
    pred <- predict(modelo, newdata = df[376:385,])
    pred[pred<0] = 0
    t4 <- Sys.time()
    results_aux <- save_metrics(prediccion = pred,
                            grown_truth = ventas.pdv$VENTA[376:385]) %>% 
      mutate(PDV = pdv, SKU = sku, ABC = abc, MODELO = "PreTDM",
             TRAIN.COST = difftime(t2, t1, units = "secs"),
             FIT.COST = difftime(t4, t3, units = "secs"))
    results[[length(results)+1]] <- results_aux
    
    ## AutoBDR ----
    
    df <- data.frame(Y = ventas.pdv$VENTA)
    if (top_modelos.pdv$trend)
      df <- cbind(df, t)
    if (top_modelos.pdv$day_freq) {
      df <- cbind(df, dias)
    } else if (top_modelos.pdv$seasonality > 1) {
      dummies <- seasonaldummy(ts(t, frequency = top_modelos.pdv$seasonality))
      df <- cbind(df, dummies)
    }
    if (top_modelos.pdv$x.reg)
      df <- cbind(df, x.reg)
    if (top_modelos.pdv$stepaic) {
      full <- lm(Y ~ ., df[1:375,])
      null <- lm(Y ~ 1, df[1:375,])
      t1 <- Sys.time()
      modelo <- step(full, scope = list(lower = null, upper = full),
                     direction = "both", trace = F)
      t2 <- Sys.time()
    } else {
      t1 <- Sys.time()
      modelo <- lm(Y ~ ., df[1:375,])
      t2 <- Sys.time()
    }
    t3 <- Sys.time()
    pred <- predict(modelo, newdata = df[376:385,])
    pred[pred<0] = 0
    t4 <- Sys.time()
    results_aux <- save_metrics(prediccion = pred,
                                grown_truth = ventas.pdv$VENTA[376:385]) %>% 
      mutate(PDV = pdv, SKU = sku, ABC = abc, MODELO = "AutoBDR",
             TRAIN.COST = difftime(t2, t1, units = "secs"),
             FIT.COST = difftime(t4, t3, units = "secs"))
    results[[length(results)+1]] <- results_aux
    
    # Estocásticos ----
    
    # ARIMA
    
    t1 <- Sys.time()
    arima <- auto.arima(ts(ventas.pdv$VENTA, frequency = 7))
    t2 <- Sys.time()
    t3 <- Sys.time()
    pred <- forecast(arima, h = 10)$mean %>% as.numeric()
    t4 <- Sys.time()
    pred[pred<0] = 0
    results_aux <- save_metrics(prediccion = pred,
                                grown_truth = ventas.pdv$VENTA[376:385]) %>% 
      mutate(PDV = pdv, SKU = sku, ABC = abc, MODELO = "ARIMA",
             TRAIN.COST = difftime(t2, t1, units = "secs"),
             FIT.COST = difftime(t4, t3, units = "secs"))
    results[[length(results)+1]] <- results_aux
    
    # rm(modelo, ventas.pdv) # Ahorro memoria
    
  }
  
  
  
  # Agregamos los outputs de todos los PDV para ese SKU
  results <- rbindlist(results) %>% as.data.frame()
  
  # rm(results_list_aux, ventas.sku, pdvs, ventas.pdv) # Ahorro memoria
  
  # Guardamos dichos outputs
  return(results)
  
}
end <- Sys.time() # Verbosity
print(end - start) # Verbosity

# Agregamos los outputs de todos los SKU
results <- rbindlist(results_list)

# Y guardamos dichos outputs en un .RData
save(results, file = "All.RData")
