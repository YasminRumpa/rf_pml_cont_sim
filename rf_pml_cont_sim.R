#install.packages(c("glmnet", "randomForest", "e1071", "data.table", "doParallel") , repos = "http://cran.us.r-project.org")
library(glmnet)
library(randomForest)
library(e1071)
library(data.table)
library(doParallel)

#Data_Generation

generate_data <- function(scenario, n = 1000, p = 10, q = 10, seed) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), ncol = p)
  Z <- matrix(rnorm(n * q), ncol = q)
  colnames(X) <- paste0("X", 1:p)
  colnames(Z) <- paste0("Z", 1:q)
  tau <- rep(NA, n)
  tau[X[,1] < 0 & X[,2] < 0] <- 1
  tau[X[,1] >= 0 & X[,2] < 0] <- 2
  tau[X[,1] < 0 & X[,2] >= 0] <- 3
  tau[X[,1] >= 0 & X[,2] >= 0] <- 4
  Z1 <- Z2 <- rep(NA, n)
  Z1[tau == 1] <- Z[tau == 1, 1]; Z2[tau == 1] <- Z[tau == 1, 2]
  Z1[tau == 2] <- Z[tau == 2, 3]; Z2[tau == 2] <- Z[tau == 2, 4]
  Z1[tau == 3] <- Z[tau == 3, 5]; Z2[tau == 3] <- Z[tau == 3, 6]
  Z1[tau == 4] <- Z[tau == 4, 7]; Z2[tau == 4] <- Z[tau == 4, 8]
  if (scenario == 1) {
    Y <- 2 * Z1 - Z2 + rnorm(n)
  } else if (scenario == 2) {
    Y <- 3 * Z1^2 + 2 * sin(Z2) + rnorm(n)
  } else if (scenario == 3) {
    Y <- 5 * sin(Z1) + 0.5 * X[,2]^3 - 2 * Z1 * Z2 + rnorm(n)
  }
  list(y = Y, x = X, z = Z)
}

# Baseline_Models

baseline_models <- function(y_train, z_train, y_test, z_test) {
  fit_lm <- lm(y_train ~ ., data = data.frame(z_train))
  pred_lm <- predict(fit_lm, newdata = data.frame(z_test))
  mse_lm <- mean((y_test - pred_lm)^2)
  
  fit_lasso <- cv.glmnet(z_train, y_train)
  pred_lasso <- predict(fit_lasso, newx = z_test, s = "lambda.min")
  mse_lasso <- mean((y_test - pred_lasso)^2)
  
  fit_rf <- randomForest(z_train, y_train)
  pred_rf <- predict(fit_rf, z_test)
  mse_rf <- mean((y_test - pred_rf)^2)
  
  fit_svm <- svm(z_train, y_train)
  pred_svm <- predict(fit_svm, z_test)
  mse_svm <- mean((y_test - pred_svm)^2)
  return(c(
    Linear = mse_lm,
    Lasso = mse_lasso,
    RandomForest = mse_rf,
    SVM = mse_svm
  ))
}

# Model_List

model_list <- list(
  linear = list(
    fit = function(y, z) lm(y ~ ., data = data.frame(y, z)),
    pred = function(fit, z) predict(fit, newdata = data.frame(z))
  ),
  lasso = list(
    fit = function(y, z) cv.glmnet(z, y),
    pred = function(fit, z) predict(fit, newx = z, s = "lambda.min")
  ),
  rf = list(
    fit = function(y, z) randomForest(z, y),
    pred = function(fit, z) predict(fit, z)
  ),
  svm = list(
    fit = function(y, z) svm(z, y),
    pred = function(fit, z) predict(fit, z)
  )
)

# Best_Model

best_model <- function(y1, z1, y2, z2) {
  mse_list <- sapply(model_list, function(f) {
    fit <- tryCatch(f$fit(y1, z1), error = function(e) NULL)
    if (is.null(fit)) return(Inf)
    y2_pred <- tryCatch(f$pred(fit, z2), error = function(e) rep(mean(y2), length(y2)))
    mean((y2 - y2_pred)^2)
  })
  name <- names(which.min(mse_list))
  fit <- model_list[[name]]$fit(y1, z1)
  list(name = name, fit = fit, pred = model_list[[name]]$pred, mse = min(mse_list, na.rm = TRUE))
}

# Predict_Tree 

predict_tree <- function(x, z, tree_res) {
  x_sel <- tree_res$x_sel
  cut_sel <- tree_res$cut_sel
  ml_sel <- tree_res$ml_sel
  internal_node <- tree_res$internal_node
  terminal_node <- tree_res$terminal_node
  
  n <- nrow(x)
  y_pred <- rep(NA, n)
  if (is.null(internal_node)) { # Terminal node only
    y_pred <- ml_sel[[1]]$pred(ml_sel[[1]]$fit, z)
  } else {
    node_hat <- rep(1, n)
    for (s in internal_node) {
      idx_s <- which(node_hat == s)
      x_s <- x[, x_sel[s]]
      cut_s <- cut_sel[s]
      lt_s <- intersect(which(x_s <= cut_s), idx_s)
      rt_s <- intersect(which(x_s > cut_s), idx_s)
      node_hat[lt_s] <- 2 * s
      node_hat[rt_s] <- 2 * s + 1
    }
    for (s in terminal_node) {
      idx_s <- which(node_hat == s)
      if (length(idx_s) > 0) {
        z_s <- z[idx_s,,drop=FALSE]
        y_pred[idx_s] <- ml_sel[[s]]$pred(ml_sel[[s]]$fit, z_s)
      }
    }
  }
  return(y_pred)
}

# Build_Tree

build_tree <- function(y1, x1, z1, p1, q_cut1, q_cut1_max, cut1,
                       node, terminal_node, internal_node, mse, ml_sel, x_sel, cut_sel, min_samples,
                       y2, x2, z2) {
  
  mse_comb <- matrix(NA, p1, q_cut1_max)
  for (j in 1:p1) {
    for (k in 1:q_cut1[j]) {
      cut1_jk <- cut1[[j]][k]
      lt1 <- which(x1[,j] <= cut1_jk)
      rt1 <- setdiff(1:nrow(x1), lt1)
      lt2 <- which(x2[,j] <= cut1_jk)
      rt2 <- setdiff(1:nrow(x2), lt2)
      n1_lt <- length(lt1)
      n1_rt <- length(rt1)
      n2_lt <- length(lt2)
      n2_rt <- length(rt2)
      cond1 <- min(n1_lt, n1_rt) > min_samples
      cond2 <- min(n2_lt, n2_rt) > 0
      if (cond1 & cond2) {
        y1_lt <- y1[lt1]; z1_lt <- z1[lt1,,drop=FALSE]
        y1_rt <- y1[rt1]; z1_rt <- z1[rt1,,drop=FALSE]
        y2_lt <- y2[lt2]; z2_lt <- z2[lt2,,drop=FALSE]
        y2_rt <- y2[rt2]; z2_rt <- z2[rt2,,drop=FALSE]
        ml_lt <- best_model(y1_lt, z1_lt, y2_lt, z2_lt)
        ml_rt <- best_model(y1_rt, z1_rt, y2_rt, z2_rt)
        y2_lt_pred <- ml_lt$pred(ml_lt$fit, z2)
        y2_rt_pred <- ml_rt$pred(ml_rt$fit, z2)
        y2_pred <- ifelse(x2[,j] <= cut1_jk, y2_lt_pred, y2_rt_pred)
        mse_comb[j,k] <- mean((y2 - y2_pred)^2)
      }
    }
  }
  mse_comb[is.na(mse_comb)] <- Inf
  mse_min <- min(mse_comb)
  if (is.infinite(mse_min)) {
    return(list(mse = mse, x_sel = x_sel, cut_sel = cut_sel, ml_sel = ml_sel,
                internal_node = internal_node, terminal_node = terminal_node))
  } else {
    mse_comb_which <- which(mse_comb == mse_min, arr.ind = TRUE)
    if (nrow(mse_comb_which) > 1)
      mse_comb_which <- mse_comb_which[sample(1:nrow(mse_comb_which), 1),]
    x_sel[node] <- mse_comb_which[1]
    cut_sel[node] <- cut1[[x_sel[node]]][mse_comb_which[2]]
    lt1 <- which(x1[,x_sel[node]] <= cut_sel[node])
    rt1 <- setdiff(1:nrow(x1), lt1)
    lt2 <- which(x2[,x_sel[node]] <= cut_sel[node])
    rt2 <- setdiff(1:nrow(x2), lt2)
    y1_lt <- y1[lt1]; x1_lt <- x1[lt1,,drop=FALSE]; z1_lt <- z1[lt1,,drop=FALSE]
    y1_rt <- y1[rt1]; x1_rt <- x1[rt1,,drop=FALSE]; z1_rt <- z1[rt1,,drop=FALSE]
    y2_lt <- y2[lt2]; x2_lt <- x2[lt2,,drop=FALSE]; z2_lt <- z2[lt2,,drop=FALSE]
    y2_rt <- y2[rt2]; x2_rt <- x2[rt2,,drop=FALSE]; z2_rt <- z2[rt2,,drop=FALSE]
    node_lt <- 2 * node
    node_rt <- 2 * node + 1
    ml_sel[[node_lt]] <- best_model(y1_lt, z1_lt, y2_lt, z2_lt)
    ml_sel[[node_rt]] <- best_model(y1_rt, z1_rt, y2_rt, z2_rt)
    internal_node <- c(internal_node, node)
    terminal_node <- setdiff(c(terminal_node, node_lt, node_rt), node)
    
    
    left_res <- build_tree(y1 = y1_lt, x1 = x1_lt, z1 = z1_lt, p1 = p1, q_cut1 = q_cut1, q_cut1_max = q_cut1_max, cut1 = cut1,
                           node = node_lt, terminal_node = terminal_node, internal_node = internal_node, mse = mse_min,
                           ml_sel = ml_sel, x_sel = x_sel, cut_sel = cut_sel, min_samples = min_samples,
                           y2 = y2_lt, x2 = x2_lt, z2 = z2_lt)
    
    
    mse <- left_res$mse
    x_sel <- left_res$x_sel
    cut_sel <- left_res$cut_sel
    ml_sel <- left_res$ml_sel
    internal_node <- left_res$internal_node
    terminal_node <- left_res$terminal_node
    
    
    right_res <- build_tree(y1 = y1_rt, x1 = x1_rt, z1 = z1_rt, p1 = p1, q_cut1 = q_cut1, q_cut1_max = q_cut1_max, cut1 = cut1,
                            node = node_rt, terminal_node = terminal_node, internal_node = internal_node, mse = mse,
                            ml_sel = ml_sel, x_sel = x_sel, cut_sel = cut_sel, min_samples = min_samples,
                            y2 = y2_rt, x2 = x2_rt, z2 = z2_rt)
    
    
    return(list(mse = right_res$mse, x_sel = right_res$x_sel, cut_sel = right_res$cut_sel,
                ml_sel = right_res$ml_sel, internal_node = right_res$internal_node, terminal_node = right_res$terminal_node))
  }
}

#rf_pml_cont
rf_pml_cont <- function(y, x, z, y_test = NULL, x_test = NULL, z_test = NULL, n_tree = 30, min_samples = 50) {
  n <- length(y)
  p <- ncol(x)
  q <- ncol(z)
  p1 <- round(p / 3)
  q1 <- round(q / 3)
  y_pred <- matrix(NA, n, n_tree)
  if (!is.null(y_test)) y_pred_test <- matrix(NA, length(y_test), n_tree)
  
  for (b in 1:n_tree) {
    message(paste("Search", b, "out of", n_tree))
    idx <- sample(1:n, n, replace = TRUE)
    oob <- setdiff(1:n, unique(idx))
    xsel <- sample(1:p, p1, replace = FALSE)
    zsel <- sample(1:q, q1, replace = FALSE)
    y1 <- y[idx]
    x1 <- x[idx, xsel, drop = FALSE]
    z1 <- z[idx, zsel, drop = FALSE]
    y2 <- y[oob]
    x2 <- x[oob, xsel, drop = FALSE]
    z2 <- z[oob, zsel, drop = FALSE]
    
    node <- 1
    terminal_node <- 1
    internal_node <- NULL
    ml_sel <- list()
    ml_sel[[node]] <- best_model(y1, z1, y2, z2)
    mse <- ml_sel[[node]]$mse
    x_sel <- NA
    cut_sel <- NA
    
    cut1 <- list()
    for (j in 1:p1) {
      cut1[[j]] <- unique(x1[,j])
      if (length(cut1[[j]]) > 9) {
        cut1[[j]] <- sort(sample(x1[,j], 10))
      }
    }
    q_cut1 <- sapply(cut1, length)
    q_cut1_max <- max(q_cut1)
    
    tree_res <- build_tree(y1 = y1, x1 = x1, z1 = z1, p1 = p1, q_cut1 = q_cut1, q_cut1_max = q_cut1_max, cut1 = cut1,
                           node = node, terminal_node = terminal_node, internal_node = internal_node, mse = mse,
                           ml_sel = ml_sel, x_sel = x_sel, cut_sel = cut_sel, min_samples = min_samples,
                           y2 = y2, x2 = x2, z2 = z2)
    
    x0 <- x[, xsel, drop = FALSE]
    z0 <- z[, zsel, drop = FALSE]
    y_pred[, b] <- predict_tree(x0, z0, tree_res)
    
    if (!is.null(y_test) && nrow(x_test) > 0 && nrow(z_test) > 0) {
      x0_test <- x_test[, xsel, drop = FALSE]
      z0_test <- z_test[, zsel, drop = FALSE]
      y_pred_test[, b] <- predict_tree(x0_test, z0_test, tree_res)
    }
  }
  
  y_pred <- apply(y_pred, 1, mean, na.rm = TRUE)
  mse <- mean((y - y_pred)^2)
  y_pred_test <- NULL
  mse_test <- NULL
  if (!is.null(y_test) && is.matrix(y_pred_test) && nrow(y_pred_test) > 0) {
    y_pred_test <- apply(y_pred_test, 1, mean, na.rm = TRUE)
    mse_test <- mean((y_test - y_pred_test)^2)
  }
  
  return(list(y_pred = y_pred, y_pred_test = y_pred_test, mse = mse, mse_test = mse_test))
}



#.........

results <- data.table()
n <- 1000
n_tree <- 30  
min_samples <- 50

for (sc in 1:3) {
  for (seed in 1:100) {  
    cat("Scenario", sc, "Seed", seed, "\n")
    data <- generate_data(scenario = sc, n = n, p = 10, q = 10, seed = seed)
    n_local <- length(data$y)
    train_idx <- sample(1:n_local, 0.7 * n_local)
    test_idx <- setdiff(1:n_local, train_idx)
    
    y_train <- data$y[train_idx]
    x_train <- data$x[train_idx, ]
    z_train <- data$z[train_idx, ]
    y_test <- data$y[test_idx]
    x_test <- data$x[test_idx, ]
    z_test <- data$z[test_idx, ]
    
    xz_train <- cbind(x_train, z_train)
    xz_test <- cbind(x_test, z_test)
    
    baseline_time <- system.time({
      mse_baseline <- baseline_models(y_train, xz_train, y_test, xz_test)
    })
    
    for (model in names(mse_baseline)) {
      results <- rbind(results, data.table(
        Scenario = sc,
        Model = model,
        Test_MSE = round(mse_baseline[[model]], 4),
        CPU_time = round(baseline_time["elapsed"], 2)
      ))
    }
    
    forest_time <- system.time({
      out <- rf_pml_cont(y = y_train, x = x_train, z = z_train, y_test = y_test, x_test = x_test, z_test = z_test,
                         n_tree = n_tree, min_samples = min_samples)
    })
    
    
    results <- rbind(results, data.table(
      Scenario = sc,
      Model = "TGML-Forest",
      Test_MSE = if (!is.null(out$mse_test)) round(out$mse_test, 4) else NA,
      CPU_time = round(forest_time["elapsed"], 2)
    ))
  }
}


summary_results <- results[, .(Avg_MSE = mean(Test_MSE, na.rm = TRUE),
                               SD_MSE = sd(Test_MSE, na.rm = TRUE),
                               Avg_CPU = mean(CPU_time, na.rm = TRUE)),
                           by = .(Scenario, Model)]
print(summary_results)

