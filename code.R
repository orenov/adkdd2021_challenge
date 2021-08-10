library(data.table)
library(lightgbm)
library(caret)
library(Rtsne)

MODEL <- 'click'
MODE  <- 'debug'

cat("===== Load Data =====")
x_train     <- fread("input/X_train.csv")
x_test      <- fread("input/X_test.csv")
y_train     <- fread("input/y_train.csv")

probs <- readRDS("precomputed_data/probs.RDS")

if (MODEL == 'click') {
  x_train[, y := y_train$click]
} else if (MODEL == 'sale') {
  x_train[, y := y_train$sale]
}
x_test[, y := NA]

if (MODE == 'submit') {
  n.train <- nrow(x_train)
  x_train <- rbind(x_train, x_test)
  x_train[, id := 1:.N]
  x_train[, probs := probs]
} else {
  x_train[, probs := probs[1:nrow(x_train)]]
}


small_train <- fread("input/small_train.csv")
dt_single   <- fread("input/aggregated_noisy_data_singles.csv")
dt_pair     <- fread("input/aggregated-noisy-data-pairs.csv")

cat("==== Prepare Features ====")

# test default for different values
actr <- function(clicks, shows, default_clicks = 10, default_ctr = 0.0993) {
  return ((clicks + default_clicks) / (shows + (default_clicks / default_ctr)))
}

dt_single[, click_rate := actr(nb_clicks, count)]
dt_single[, sale_rate  := actr(nb_sales, count, default_ctr = 0.004010095)]

dt_pair[feature_1_id > feature_2_id,
        c("feature_1_id", "feature_2_id", "feature_1_value", "feature_2_value") := list(feature_2_id, feature_1_id, feature_2_value, feature_1_value)]

dt_pair <- merge(dt_pair, dt_single[, .(feature_1_value, feature_1_id, click_rate, sale_rate, nb_clicks, nb_sales)],
                 by.x = c('feature_1_value', 'feature_1_id'),
                 by.y = c('feature_1_value', 'feature_1_id'))

setnames(dt_pair, c('click_rate', 'sale_rate', 'nb_clicks.x', 'nb_sales.x', 'nb_clicks.y', 'nb_sales.y'), 
         c('click_rate_1', 'sale_rate_1', 'nb_clicks', 'nb_sales', 'nb_clicks_1', 'nb_sales_1'))

dt_pair <- merge(dt_pair, dt_single[, .(feature_1_value, feature_1_id, click_rate, sale_rate,  nb_clicks, nb_sales)],
                 by.x = c('feature_2_value', 'feature_2_id'),
                 by.y = c('feature_1_value', 'feature_1_id'))

setnames(dt_pair, c('click_rate', 'sale_rate', 'nb_clicks.x', 'nb_sales.x', 'nb_clicks.y', 'nb_sales.y'), 
         c('click_rate_2', 'sale_rate_2', 'nb_clicks', 'nb_sales', 'nb_clicks_2', 'nb_sales_2'))

# here can be issues with negative sqrt
dt_pair[, sale_rate  := (abs(sale_rate_1) * log1p(abs(nb_sales_1)) + abs(sale_rate_2) * log1p(abs(nb_sales_2))) / (log1p(abs(nb_sales_2)) + log1p(abs(nb_sales_1))) ]
dt_pair[, click_rate  := (abs(click_rate_1) * log1p(abs(nb_clicks_1)) + abs(click_rate_2) * log1p(abs(nb_clicks_2))) / (log1p(abs(nb_clicks_2)) + log1p(abs(nb_clicks_1))) ]

dt_pair[, sale_rate  := actr(nb_sales,  count, default_ctr = sale_rate)]
dt_pair[, click_rate := actr(nb_clicks, count, default_ctr = click_rate)]

dt_pair[, c('click_rate_1', 'click_rate_2', 'sale_rate_1', 'sale_rate_2', 'nb_clicks_1', 'nb_sales_1', 'nb_clicks_2', 'nb_sales_2') := NULL]

x_train[, sum_click_rate := 0.0]
x_train[, sum_sale_rate  := 0.0]
x_train[, sum_click_counts := 0.0]

x_train[, noisy_clicks := 0.0]
x_train[, noisy_sales  := 0.0]

for (i in 0:18) {
  x_train <- merge(x_train, dt_single[feature_1_id == i], by.x = paste0("hash_",  i, collapse = ''), by.y = 'feature_1_value', all.x = TRUE)
  x_train[["feature_1_value"]] <- NULL
  x_train[["feature_1_id"]] <- NULL
  nn <- c('click_rate',  'sale_rate', 'count', 'nb_clicks', 'nb_sales')
  
  x_train[, noisy_clicks := noisy_clicks + 1 * (nb_clicks < 10)]
  x_train[, noisy_sales  := noisy_sales  + 1 * (nb_sales < 10)]
  
  x_train[is.na(click_rate), click_rate := 0.0993]
  x_train[is.na(sale_rate), sale_rate := 0.004010095]
  x_train[is.na(count), count := 0]
  x_train[, sum_click_rate := sum_click_rate + click_rate]
  x_train[, sum_sale_rate  := sum_sale_rate + sale_rate]
  x_train[, sum_click_counts := sum_click_counts + count]
  
  setnames(x_train, nn, paste0(nn, '_', i))
}



tli  <- c(0, 1, 2, 3, 10, 12, 13, 14, 16, 17)
rest <- c(0:18)
tli  <- c(0:18)

x_train[, sum_click_rate_pair := 0.0]
x_train[, sum_sale_rate_pair  := 0.0]


for (i in tli) {
  print(i)
  for (j in rest) {
    if (i == j) next
    if (j %in% tli & j < i) next
    name1 <- paste0("hash_",  i, collapse = '')
    name2 <- paste0("hash_",  j, collapse = '')
    x_train <- merge(x_train,
                     dt_pair[feature_1_id == i][feature_2_id ==  j], 
                     by.x = c(name1, name2), 
                     by.y = c('feature_1_value', 'feature_2_value'),
                     all.x = TRUE)
    x_train[['feature_1_id']] <- NULL
    x_train[['feature_2_id']] <- NULL
    x_train[['nb_clicks']]    <- NULL
    x_train[['nb_sales']]     <- NULL
    
    x_train[is.na(click_rate), click_rate := 0.0993]
    x_train[is.na(sale_rate), sale_rate   := 0.004010095]
    x_train[is.na(count), count := 0]
    x_train[, sum_click_rate_pair := sum_click_rate_pair + click_rate]
    x_train[, sum_sale_rate_pair  := sum_sale_rate_pair + sale_rate]
    
    if (MODEL == 'sale') {
      nn <- c('click_rate', 'sale_rate',  'count')
    } else {
      nn <- c('click_rate', 'count')
      x_train[, sale_rate := NULL]
    }
    
    
    
    setnames(x_train, nn, paste0(nn, '_', i, '_', j))
  }
}

#saveRDS(x_train,   paste0("precomputed_data/rates_", MODEL, "_", MODE, ".RDS"))
#x_train <- readRDS(paste0("precomputed_data/rates_", MODEL, "_", MODE, ".RDS"))
# PCA
number_of_components = 40
pca_features <- names(x_train)[grepl("click_rate_", names(x_train))]
pca <- prcomp(subset(x_train, select = pca_features), center = T, scale = T)
pca_vals <- as.data.table(as.matrix(subset(x_train, select = pca_features)) %*%  as.matrix(pca$rotation[, 1:number_of_components]))
setnames(pca_vals, paste0(names(pca_vals), '_click'))
x_train <- cbind(x_train, pca_vals)

pca_features <- names(x_train)[grepl("count_", names(x_train))]

for (f in pca_features) {
  if (sum(x_train[[f]]) == 0) {
    x_train[[f]] <- NULL
  }
}

if (MODEL == 'sale') {
  pca_features <- names(x_train)[grepl("sale_rate_", names(x_train))]
  pca <- prcomp(subset(x_train, select = pca_features), center = T, scale = T)
  pca_vals <- as.data.table(as.matrix(subset(x_train, select = pca_features)) %*%  as.matrix(pca$rotation[, 1:number_of_components]))
  setnames(pca_vals, paste0(names(pca_vals), '_sale'))
  x_train <- cbind(x_train, pca_vals)
}

if (FALSE) {
  #tsne
  if (file.exists("precomputed_data/tsne_data_click.RDS")) {
    tsne_data <- readRDS("precomputed_data/tsne_data_click.RDS")[1:nrow(x_train)]
  } else {
    tsne_features <- names(x_train)[grepl("click_rate_", names(x_train))]
    system.time(tsne_out <- Rtsne(data.matrix(subset(x_train, select = tsne_features)),
                                  max_iter = 500, verbose=TRUE,
                                  dims = 3,
                                  num_threads = 5,
                                  check_duplicates = FALSE))
    
    tsne_data <- as.data.table(data.matrix(tsne_out$Y))
    setnames(tsne_data, paste0("TSNE_", names(tsne_data)))
  }
  x_train[, tsne_1 := tsne_data$TSNE_V1]
  x_train[, tsne_2 := tsne_data$TSNE_V2]
  x_train[, tsne_3 := tsne_data$TSNE_V3]
  
  saveRDS(tsne_data, "precomputed_data/tsne_data_click.RDS")
}

hash_names <- c()
for (i in 0:18) {
  name <- paste0("hash_",  i, collapse = '')
  hash_names <- c(hash_names, name)
  print(name)
  print(uniqueN(x_train[[name]]))
  x_train[[name]] <- NULL
}

rm(dt_pair)
rm(dt_single)


cat("====== Prepare data for model ======")
if (MODE == 'submit') {
  x_test <- x_train[is.na(y)][order(id)]
  x_test[, id := NULL]
  x_test[, y := NULL]
  dtest <- as.matrix(x_test)
  x_train[, id := NULL]
}
x_train <- x_train[!is.na(y)]

saveRDS(x_train, file = 'precomputed_data/train.RDS')


cat("==== Train And Submit Phase ====")


p <- list(boosting_type = "gbdt", 
          objective = "binary",
          metric = "binary_logloss", 
          nthread = 15, 
          learning_rate = 0.01, 
          max_depth = 5,
          num_leaves = 190,
          feature_fraction = 0.5, # possible increase can improve the model
          bagging_fraction = 0.9,
          bagging_freq = 1,
          max_bin = 1000,
          verbosity = -1,
          lambda_l1 = 0.1,
          lambda_l2 = 0.1,
          extra_trees = T,
          min_data_in_leaf = 600 # 0.2348867 - max_depth  = 5, learning_rate = 0.01
)

set.seed(42)

if (MODE == 'submit') {
  split <- createFolds(as.factor(x_train$y), 10)
  
  response <- x_train$y
  x_train[, y := NULL]
  x_train <- as.matrix(x_train)
  for(i in 1:10) {
    dtrain <- lgb.Dataset(data = x_train[-split[[i]],], label=response[-split[[i]]])
    dval   <- lgb.Dataset(data = x_train[split[[i]],], label=response[split[[i]]])
    model  <- lgb.train(p, dtrain, 10000, valids = list(val=dval), early_stopping_rounds = 50, eval_freq = 100)
    
    pred <- predict(model, dtest)
    if (i==1) temp <- pred else temp <- temp + pred
  }
  
  submit <- data.table(prediction = temp / 10)
  write.table(x = submit, file = paste0("submissions/y_hat_", MODEL, ".txt", collapse=""), quote = FALSE, col.names = FALSE, row.names = FALSE)
} else if (MODE == 'debug') {
  response <- x_train$y
  x_train[, y := NULL]
  dtrain <- lgb.Dataset(data=as.matrix(x_train), label=response)
  clf.cv <- lgb.cv(params = p, nrounds = 20000, data = dtrain, nfold = 5, early_stopping_rounds=50, eval_freq = 100, seed = 1990)
  print(clf.cv)
  clf <- lgb.train(params = p, nrounds = clf.cv$best_iter, data = dtrain)
  imp <- lgb.importance(model = clf)
  print(clf.cv$best_score)
}

# CV: init10.R
# CLICK loss = 0.231725
# SALE loss  = 0.019942
# Public:
# CLICK loss = 0.230671
# SALE loss  = 0.018032
