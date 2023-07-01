options(scipen = 999)

library(doParallel)
library(data.table)

datos <- tryCatch(
  datos <- fread("datos_anonimos.csv"),
  error = function(e) {
    message("Datos aleatorios generados")
    datos <- expand.grid(PDV = paste0("PDV_",1:3000),
                         SKU = paste0("SKU_",1:5))
    datos <- cbind(datos,
                   VENTAS = rnorm(nrow(datos), 10, 3))
    return(datos)
  }
)
datos$VENTAS[datos$VENTAS < 0] = 0

save_result <- function(stats, predicciones) {
  results <- list(
    stats = stats,
    predicciones = predicciones
  )
  class(results) <- append(class(results), "resultado_forecast")
  return(results)
}

skus <- unique(SO$SKU)

closeAllConnections()
num_cores <- min(length(skus), detectCores() - 2)
parallelCluster <- makeCluster(num_cores, type = "SOCK"); # type = "FORK" para Ticana
setDefaultCluster(parallelCluster);
registerDoParallel(parallelCluster);
showConnections()

start <- Sys.time()
final_result_list <- foreach(sku = skus, .inorder = FALSE,
                             .packages = c("tidyr", "tidyselect", "tseries",
                                           "forecast", "data.table", "RSNNS")) %dopar% {
  
  SO.cut <- SO[SO$SKU==sku,]
  mercado = unique(SO.cut$MERCADO)
  
  stats_list <- list()
  preds_list <- list()
  
  # Sin escalar
  stacked <- SO.cut[,c(1,2,5,6)] %>%
    pivot_wider(names_from = FECHA, values_from = SO) # formato mlp-friendly
  info.pos <- stacked[,1:2];
  stacked <- stacked[,-(1:2)]
  
  for (segmento in unique(info.pos$ABC)) {
    
    info.pos.seg <- info.pos[info.pos$ABC == segmento,]
    
    # REDES (TRAIN) ----
    
    stacked.seg <- stacked[info.pos$ABC == segmento,]
    stacked_scaled.seg <- apply(stacked.seg, 1, escalar) %>% t()
    
    ## MLP ----
    
    set.seed(2023)
    t1 = Sys.time()
    mlp <- mlp(stacked_scaled.seg[,1:366], # input 366 dias
               stacked_scaled.seg[,367:376], # output 10 dias
               size = 10, learnFuncParams = c(0.01)) # parametros
    mlp_entrenada <- train(mlp, stacked_scaled.seg[,1:366], stacked_scaled.seg[,367:376], maxit = 100, learnFunc = "Std_Backpropagation")
    t2 = Sys.time()
    
    mlp_cost <- as.numeric(difftime(t2, t1, units = "secs")) / nrow(stacked_scaled.seg)
    
    ## ELMAN ----
    
    set.seed(2023)
    t1 = Sys.time()
    elman <- elman(stacked_scaled.seg[,1:366], # input 366 dias
                   stacked_scaled.seg[,367:376], # output 10 dias
                   size = 10, learnFuncParams = c(0.01)) # parametros
    elman_entrenada <- train(elman, stacked_scaled.seg[,1:366], stacked_scaled.seg[,367:376], maxit = 100, learnFunc = "Std_Backpropagation")
    t2 = Sys.time()
    
    elman_cost <- as.numeric(difftime(t2, t1, units = "secs")) / nrow(stacked_scaled.seg)
    
    for (pos in info.pos.seg$POS) {
      
      SO.pos <- SO.cut[SO.cut$POS == pos,]
      SO.pos <- SO.pos[11:386,] # los 10 dias anteriores solo los queriamos para la red neuronal
      serie <- ts(SO.pos$SO, frequency = 7)
      gt <- as.numeric(SO.pos$SO[367:376])
      tipo <- ifelse(all(gt == 0), "TestConstant", "Normal")
      
      # SERIES CONSTANTES ----
      # Si la serie es constante en el conjunto train, no se utilizaran los algoritmos para ahorrar tiempo
      # Ademas, se les indicara MODELO = "none" para excluirlos de los estudios
      if (all(SO.pos$SO[1:366] == 0)) { # series constantes
        
        pred <- rep(0,10)
        
        predicciones <- data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = as.numeric(sku), MODELO = "none", REAL = gt, PRED = pred)
        
        error <- (gt - pred)
        rmse <- sqrt(mean(error^2))
        acum.error <- sum(pred) - sum(gt)
        relat.error <- acum.error / sum(gt)
        
        stats <- data.frame(MERCADO = mercado, SKU = as.numeric(sku), POS = pos, ABC = segmento, MODELO = "none",
                            RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                            TRAIN.COST = 0, TEST.COST = 0, TIPO = "TrainConstant")
        
        stats_list[[length(stats_list)+1]] <- stats
        preds_list[[length(preds_list)+1]] <- predicciones
        
        next
        
      }
      
      # DETERMINISTICOS ----
      
      dummies <- seasonaldummy(serie)
      t <- time(serie)
      train <- data.frame(SO = serie[1:366], t = t[1:366]) %>% cbind(dummies[1:366,])
      test <- data.frame(SO = serie[367:376], t = t[367:376]) %>% cbind(dummies[367:376,])
      
      ## TENDENCIA ----
      
      # modelizacion
      t1 = Sys.time()
      trend <- lm(SO ~ t, data = train)
      t2 = Sys.time()
      
      # ajuste
      t3 = Sys.time()
      pred <- predict(trend, newdata = data.frame(t = test$t))
      pred[pred<0]=0
      t4 = Sys.time()
      
      predicciones <- data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = sku, MODELO = "trend", REAL = gt, PRED = pred)
      
      # errores
      error <- as.numeric(gt - pred)
      rmse <- sqrt(mean(error^2))
      acum.error <- sum(pred) - sum(gt)
      relat.error <- acum.error / sum(gt)
      
      stats <- data.frame(MERCADO = mercado, SKU = sku, POS = pos, ABC = segmento, MODELO = "trend",
                          RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                          TRAIN.COST = as.numeric(difftime(t2, t1, units = "secs")),
                          TEST.COST = as.numeric(difftime(t4, t3, units = "secs")), TIPO = tipo)
      
      ## ESTACIONALIDAD ----
      
      # modelizacion
      t1 = Sys.time()
      seas <- lm(SO ~ S1 + S2 + S3 + S4 + S5 + S6, data = train)
      t2 = Sys.time()
      
      # ajuste
      t3 = Sys.time()
      pred <- predict(seas, newdata = test[,3:8])
      pred[pred<0]=0
      t4 = Sys.time()
      
      predicciones <- rbindlist(list(predicciones,
                                     data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = sku, MODELO = "seas", REAL = gt, PRED = pred)))
      
      # errores
      error <- as.numeric(gt - pred)
      rmse <- sqrt(mean(error^2))
      acum.error <- sum(pred) - sum(gt)
      relat.error <- acum.error / sum(gt)
      
      stats <- rbindlist(list(stats,
                              data.frame(MERCADO = mercado, SKU = sku, POS = pos, ABC = segmento, MODELO = "seas",
                                         RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                                         TRAIN.COST = as.numeric(difftime(t2, t1, units = "secs")),
                                         TEST.COST = as.numeric(difftime(t4, t3, units = "secs")), TIPO = tipo)))
      
      ## TENDENCIA Y ESTACIONALIDAD ----
      
      # modelizacion
      t1 = Sys.time()
      trend.seas <- lm(SO ~ ., data = train)
      t2 = Sys.time()
      
      # ajuste
      t3 = Sys.time()
      pred <- predict(trend.seas, newdata = test[,2:8])
      pred[pred<0]=0
      t4 = Sys.time()
      predicciones <- rbindlist(list(predicciones,
                                     data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = sku, MODELO = "trend&seas", REAL = gt, PRED = pred)))
      
      # errores
      error <- as.numeric(gt - pred)
      rmse <- sqrt(mean(error^2))
      acum.error <- sum(pred) - sum(gt)
      relat.error <- acum.error / sum(gt)
      
      stats <- rbindlist(list(stats,
                              data.frame(MERCADO = mercado, SKU = sku, POS = pos, ABC = segmento, MODELO = "trend&seas",
                                         RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                                         TRAIN.COST = as.numeric(difftime(t2, t1, units = "secs")),
                                         TEST.COST = as.numeric(difftime(t4, t3, units = "secs")), TIPO = tipo)))
      
      # ARIMA ----
      
      # modelizacion
      t1 = Sys.time()
      sarima <- auto.arima(serie[1:366], seasonal = TRUE, nmodels = 30, approximation = T, stepwise = T) %>% suppressWarnings()
      t2 = Sys.time()
      
      # ajuste
      t3 = Sys.time()
      pred <- forecast(sarima, h = 10)
      pred <- as.data.frame(pred)$`Point Forecast`
      pred[pred<0]=0
      t4 = Sys.time()
      
      predicciones <- rbindlist(list(predicciones,
                                     data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = sku, MODELO = "sarima", REAL = gt, PRED = pred)))
      
      # errores
      error <- (gt - pred)
      rmse <- sqrt(mean(error^2))
      acum.error <- sum(pred) - sum(gt)
      relat.error <- acum.error / sum(gt)
      
      stats <- rbindlist(list(stats,
                              data.frame(MERCADO = mercado, SKU = sku, POS = pos, ABC = segmento, MODELO = "sarima",
                                         RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                                         TRAIN.COST = as.numeric(difftime(t2, t1, units = "secs")),
                                         TEST.COST = as.numeric(difftime(t4, t3, units = "secs")), TIPO = tipo)))
      
      # REDES (TEST) ----
      
      
      
      ## MLP ----
      
      t3 = Sys.time()
      pred <- predict(mlp_entrenada, t(stacked_scaled.seg[which(info.pos.seg$POS == pos), 11:376])) %>% as.numeric()
      pred[pred<0]=0
      t4 = Sys.time()
      
      pred <- desescalar(z = pred,
                         maximo = max(stacked_scaled.seg[which(info.pos.seg$POS == pos), 1:376]),
                         minimo = min(stacked_scaled.seg[which(info.pos.seg$POS == pos), 1:376]))
      
      predicciones <- rbindlist(list(predicciones,
                                     data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = sku, MODELO = "mlp",
                                                REAL = gt, PRED = pred)))
      
      error <- (gt - pred) %>% as.numeric()
      rmse <- sqrt(mean(error^2))
      acum.error <- sum(pred) - sum(gt)
      relat.error <- acum.error / sum(gt)
      
      stats <- rbindlist(list(stats,
                              data.frame(MERCADO = mercado, SKU = sku, POS = pos, ABC = segmento, MODELO = "mlp",
                                         RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                                         TRAIN.COST = mlp_cost, TEST.COST = as.numeric(difftime(t4, t3, units = "secs")), TIPO = tipo)))
      
      ## ELMAN ----
      
      t3 = Sys.time()
      pred <- predict(elman_entrenada, t(stacked_scaled.seg[which(info.pos.seg$POS == pos), 11:376])) %>% as.numeric()
      pred[pred<0]=0
      t4 = Sys.time()
      
      pred <- desescalar(z = pred,
                         maximo = max(stacked_scaled.seg[which(info.pos.seg$POS == pos), 1:376]),
                         minimo = min(stacked_scaled.seg[which(info.pos.seg$POS == pos), 1:376]))
      
      predicciones <- rbindlist(list(predicciones,
                                     data.frame(POSUNIQUECODE = pos, ABC = segmento, MERCADO = mercado, SKU = sku, MODELO = "elman",
                                                REAL = gt, PRED = pred)))
      
      error <- (gt - pred) %>% as.numeric()
      rmse <- sqrt(mean(error^2))
      acum.error <- sum(pred) - sum(gt)
      relat.error <- acum.error / sum(gt)
      
      stats <- rbindlist(list(stats,
                              data.frame(MERCADO = mercado, SKU = sku, POS = pos, ABC = segmento, MODELO = "elman",
                                         RMSE = rmse, ACUM.ERROR = acum.error, RELAT.ERROR = relat.error, ERROR.DIAS = NA,
                                         TRAIN.COST = elman_cost, TEST.COST = as.numeric(difftime(t4, t3, units = "secs")), TIPO = tipo)))
      
      # SAVE RESULTS ----
      
      stats_list[[length(stats_list)+1]] <- stats
      preds_list[[length(preds_list)+1]] <- predicciones
      
    }
    
  }
  
  stats <- rbindlist(stats_list)
  predicciones <- rbindlist(preds_list)
  
  resultado_sku <- save_result(stats = stats, predicciones = predicciones)
  
  return(resultado_sku)
}

stopCluster(parallelCluster);

output <- data.frame()
forecast <- data.frame()
for (i in 1:length(final_result_list)) {
  output <- rbindlist(list(output, final_result_list[[i]]$stats))
  forecast <- rbindlist(list(forecast, final_result_list[[i]]$predicciones))
}

end <- Sys.time()
print(end - start)
