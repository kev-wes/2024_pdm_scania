library(tidyverse)
library(mlr)

### Data Acquisition

# Read train set
aps_train <-
  read.csv("./aps_failure_training_set.csv")
# Put identifier at end of train set
aps_train$set <- "train"
# Read test set
aps_test <-
  read.csv("./aps_failure_test_set.csv")
# Put identifier at end of test set
aps_test$set <- "test"
aps_full = rbind(aps_train, aps_test)
# Replace "na" strings with R NA values
aps_full[aps_full == "na"] <- NA
# Typecast sensor values to numeric
aps_full[, 2:(ncol(aps_full) - 1)] = apply(aps_full[, 2:(ncol(aps_full) -
                                                           1)], 2, as.numeric)

### Data Exploration

# How many failures related and unrelated to air pressure systems (APS) exist?
table(aps_full$class)

# Summary
summary(aps_full)


### Data Preparation

##################### Exercise 5 Optimization: SOF Remove constant values (there are none)
aps_full = aps_full[, !apply(aps_full, 2, function(x)
  max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
##################### Exercise 5 Optimization: SOF Remove constant values (there are none)

##################### Exercise 5 Optimization: SOF Remove collinearities
cor = cor(aps_full[, 2:170], use = "complete.obs")
diag(cor) = NA # remove autocorrelation
cor = abs(cor) # absolutes
high_cor = which(cor > 0.95, arr.ind = TRUE)
aps_full = aps_full[,-(unique(high_cor[, 1]) + 1)]
##################### Exercise 5 Optimization: SOF Remove collinearities

# Exercise 1: Normalization
aps_norm = aps_full
normalize_fun <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}
aps_norm[, 2:(ncol(aps_norm) - 1)] = as.data.frame(apply(aps_full[, 2:(ncol(aps_norm) -
                                                                         1)], 2, normalize_fun))

# Exercise 2: Imputation
aps_imp = aps_norm
na_per_feature = colMeans(is.na(aps_imp[, 2:(ncol(aps_imp) - 1)])) #Percentages
aps_imp = aps_imp[, -as.numeric(which(na_per_feature > 0.05) + 1)] # Remove columns w/ NA > 0.05
aps_imp = aps_imp %>%
  mutate_if(is.numeric, function(x)
    ifelse(is.na(x), median(x, na.rm = T), x)) # Replace with median

### Fault Detection

# Exercise 3: Classification
# Make task
task = makeClassifTask(data = aps_imp[, -ncol(aps_imp)], target = "class")
# Split data again
train_set = which(aps_imp$set == "train")
test_set = which(aps_imp$set == "test")

##################### Exercise 5 Optimization: SOF Class balancing, e.g., random undersampling (do not use together with thresholding!)
train_set = c(which(aps_imp$set == "train" &
                      aps_imp$class == "pos"),
              sample(which(
                aps_imp$set == "train" & aps_imp$class == "neg"
              ), 1000))
##################### Exercise 5 Optimization: SOF Class balancing, e.g., random undersampling (do not use together with thresholding!)

# Make learner
##################### Exercise 5 Optimization: SOF Hyperparameter tuning
treeParamSpace = makeParamSet(
  # Create grid
  makeIntegerParam("minsplit", lower = 5, upper = 20),
  makeIntegerParam("minbucket", lower = 3, upper = 10),
  makeNumericParam("cp", lower = 0.01, upper = 0.1),
  makeIntegerParam("maxdepth", lower = 3, upper = 10)
)
randSearch = makeTuneControlRandom(maxit = 20) # random search
cvForTuning <-
  makeResampleDesc("CV", iters = 5) # Define 5-fold cross validation
res = tuneParams(
  "classif.rpart",
  # Tune!
  task = subsetTask(task, subset = train_set),
  resampling = cvForTuning,
  par.set = treeParamSpace,
  control = randSearch
)
lrn = setHyperPars(
  makeLearner("classif.rpart", predict.type = "prob"),
  # Set hyperparam of best config
  minsplit = res$x$minsplit,
  minbucket = res$x$minbucket,
  cp = res$x$cp,
  maxdepth = res$x$maxdepth
)
##################### 5 Optimization: EOF Hyperparameter tuning

# Train
model = train(lrn, task, subset = train_set)
# Predict
pred = predict(model, task = task, subset = test_set)

# Exercise 4: Evaluation
scania_cost = matrix(c(0, 500, 10, 0), ncol = 2)
rownames(scania_cost) = colnames(scania_cost) = getTaskClassLevels(task)
# Encapsulate the cost matrix in a Measure object
scania_cost_msr = makeCostMeasure(
  id = "scania_cost_msr",
  name = "Scania Costs",
  costs = scania_cost,
  combine = sum,
  minimize = TRUE,
  best = 0,
  worst = 500
)
performance(pred, measures = scania_cost_msr) # Evaluation
calculateConfusionMatrix(pred) # Confusion matrix

##################### Exercise 5 Optimization: SOF Threshold (do not use together with undersampling!)
th = scania_cost[2, 1] / (scania_cost[2, 1] + scania_cost[1, 2])
pred_th = setThreshold(pred, th)
performance(pred_th, measures = scania_cost_msr) # Evaluation
calculateConfusionMatrix(pred_th) # Confusion matrix
##################### Exercise 5 Optimization: EOF Threshold (do not use together with undersampling!)
