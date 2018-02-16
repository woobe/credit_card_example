
# Simple For loop to try different threshold

# min
a = quantile(d_recon_error_train[d_recon_error_train$Class == 0,]$Reconstruction.MSE,
             probs = 0.75)
# max
b = quantile(d_recon_error_train[d_recon_error_train$Class == 1,]$Reconstruction.MSE,
             probs = 0.25)

# define prob
for (n_prob in c(0, 0.1, 0.5, 0.9, 1)) {

  # define threshold
  threshold = ((b - a) * n_prob) + a
  cat("\nProbs cutoff:", n_prob, "... Threshold:", threshold, "\n")

  # Split data
  row_train_exp = which(d_recon_error_train$Reconstruction.MSE < threshold)
  row_train_ano = which(d_recon_error_train$Reconstruction.MSE >= threshold)

  # Split training data
  h_train_exp = h_train[row_train_exp,]
  h_train_ano = h_train[row_train_ano,]

  # Train same baseline random forest model with group "Expected"
  model_exp = h2o.randomForest(x = features,
                               y = "Class",
                               training_frame = h_train_exp,
                               validation_frame = h_valid,
                               model_id = "expected",
                               seed = 1234)

  # Train same baseline random forest model with group "Anomalies"
  model_ano = h2o.randomForest(x = features,
                               y = "Class",
                               training_frame = h_train_ano,
                               validation_frame = h_valid,
                               model_id = "anomalies",
                               seed = 1234)

  # Bag the two models for holdout
  yhat_holdout_exp = h2o.predict(model_exp, h_holdout)
  yhat_holdout_ano = h2o.predict(model_ano, h_holdout)
  yhat_holdout_bag = data.frame(exp = as.data.frame(yhat_holdout_exp$p1),
                                ano = as.data.frame(yhat_holdout_ano$p1))
  yhat_holdout_bag$avg = rowMeans(yhat_holdout_bag)

  # Evaluate performance
  library(Metrics)
  d_eval_holdout = data.frame(as.data.frame(h_holdout$Class),
                              predicted = yhat_holdout_bag$avg)
  auc_holdout = Metrics::auc(d_eval_holdout$Class, d_eval_holdout$predicted)

  cat("AUC Holdout (Baseline):", h2o.auc(h2o.performance(model_baseline, h_holdout)), "\n")
  cat("AUC Holdout (Bagging):", auc_holdout, "\n")


}




