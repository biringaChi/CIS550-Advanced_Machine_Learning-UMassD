# author = biringaChi
# email = biringachidera@gmail.com

# This file contains code for the exploration of regression models

packages = c("tidyverse","R.matlab","caret","glmnet")
verify_packages <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }
})

# specify file path otherwise
file_path <- "Abalone.mat"

data_handling <- function(file_path, dataset) {
  # Reads and processes matlab data format
  if(!is.null(file_path)) dataset <- readMat(file_path)
  else warning("Incorrect file path", call. = TRUE)
  process <- function(dataset, V901) {
    input.features <<- as.matrix(dataset[["instance.matrix"]])
    output.ages <<- as.matrix(dataset[["label.vector"]])
  }
  process(dataset)
  return(as.data.frame(cbind(input.features, output.ages)))
}

data_split <- function(data) {
  # splits data into training and testing set
 abalone <-  data_handling(file_path, dataset)
  set.seed(1234)      
  splits <- sample(1:2, size = nrow(abalone), prob = c(.7, .3), replace = T)
  training <<- abalone[splits == 1,]
  testing <<- abalone[splits == 2,]
}
data_split(abalone)

cv <- function() {
  # Computes Cross validation
  set.seed(1234)
  cv <- trainControl(method = "repeatedcv", number = 10,
                     repeats = 5, verboseIter = T)
  return(cv)
}

#========================= Linear regression ==================================#
linear_model <- function(V9, ., training) {
  # Computes Linear regression model
  set.seed(1234)
  lm <- train(V9 ~ ., data = training, method = 'lm', trControl = cv())
  return(lm)
}

lm <- linear_model(V9, ., training)

predictions <- function(model, test) {
  # Computes Predictions
  p <- predict(model, test)
  return(p)
}

average_square_error <- function(model) {
  # Computes Average Square-Error
  predictions <- predictions(model, testing)
  ase <- mean((predictions - testing$V9)^2)
  result <- sprintf("Average Square-Error = %.3f", ase)
  return(result)
}

average_square_error(lm)
#======================= End of Linear Regression =============================#

#============================== Lasso  ========================================#
x_train <- as.matrix(training[,-9])
y_train <- as.matrix(training$V9)
x_test <- as.matrix(testing[,-9])
y_test <- as.matrix(testing$V9)
grid <- 10^seq(from = -4, to = 1, length.out = 100)

compute_lasso <- function(x_train, y_train, grid, x_test) {
  # Computes lasso 
  set.seed(1234)
  lasso <<- cv.glmnet(x_train, y_train, alpha = 1, lambda = grid)
  opt_lambda_lasso <<- lasso$lambda.1se
  predictions <- predict(lasso, s = opt_lambda_lasso, newx = x_test)
  return(predictions)
}

lasso_preds <- compute_lasso(x_train, y_train, grid, x_test)

lasso_ase <- function(y_test) {
  ase <- mean((lasso_preds - y_test)^2)
  result <- sprintf("Lasso Regression Average Square-Error = %.3f", ase)
  return(result)
}
lasso_ase(y_test)
#========================== End of Lasso  =====================================#

#========================== Ridge Regression ==================================#
compute_ridge <- function(x_train, y_train, grid, x_test) {
  set.seed(1234)
  ridge <<- cv.glmnet(x_train, y_train, alpha = 0 , lambda = grid)
  opt_lambda_ridge <<- ridge$lambda.1se
  predictions <- predict(ridge, s = opt_lambda_ridge, newx = x_test)
  return(predictions)
}
ridge_preds <- compute_ridge(x_train, y_train, grid, x_test)

ridge_ase <- function(y_test) {
  ase <- mean((ridge_preds - y_test)^2)
  result <- sprintf("Ridge Regression - Average Square-Error = %.3f", ase)
  return(result)
}
ridge_ase(y_test)
#========================== End of Ridge Regression ===========================#

#========================= Visualization - Lasso ==============================#
compute_lasso_df <- function(lasso, opt_lambda_lasso) {
  # Computes coefficient extractions for individual lambdas
  coeff <- predict(lasso, type = "coefficients", s = opt_lambda_lasso)[1:100]
  lasso_df <- data.frame(lambda = lasso$lambda, coefficients = coeff, 
                         stringsAsFactors = FALSE)
}
lasso_df <- compute_lasso_df(lasso, opt_lambda_lasso)

ggplot(lasso_df[1:20,], aes(x = lambda, y = coefficients)) + theme_bw() +
  geom_bar(stat="Identity") + labs(y = "Coefficients(β)", x = "Lambda(λ)", 
                                title = "Lasso - Coefficieints(β) vs Lambda(λ)")
#======================== End of visualization - Lasso ========================#

#========================= Visualization - Ridge ==============================#
compute_ridge_df <- function(ridge, opt_lambda_ridge) {
  # Computes coefficient extractions for individual lambdas
  coeff <- predict(ridge, type = "coefficients", s = opt_lambda_ridge)[1:100]
  ridge_df <- data.frame(lambda = ridge$lambda, coefficients = coeff, 
                         stringsAsFactors = FALSE)
}
ridge_df <- compute_ridge_df(ridge, opt_lambda_ridge)

ggplot(ridge_df[1:20,], aes(x = lambda, y = coefficients)) + theme_bw() +
  geom_bar(stat="Identity") + labs(y = "Coefficients(β)", x = "Lambda(λ)", 
                                title = "Ridge - Coefficieints(β) vs Lambda(λ)")
