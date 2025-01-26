############################################################ First steps

# Load the necessary libraries
install.packages("dplyr")
install.packages("tidyr")
library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(corrplot)

# Import dataset
data <- read.csv("h:/Desktop/BSE/data science/hfi_cc_2022.csv")

# Display first lines of dataset
head(data)
dim(data) 
nrow(data)
ncol(data)
colnames(data)


############################################################ Graphics to understand the dataset as a whole

#Global average evolution index of freedom
# Calculate the annual average of hf_score
year_average  <- data %>%
  group_by(year) %>%
  summarize(average_hf_score = mean(hf_score, na.rm = TRUE))
# Create an online graph of the evolution of the average hf_score per year
ggplot(year_average, aes(x = year, y = average_hf_score )) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  theme_minimal() +
  labs(title = "Annual Evolution of the Average HF Score Indicator",
       x = "Year",
       y = "Average HF Score",
       caption = "Source: Human Freedom Index")

# Histogram of human freedom scores
ggplot(data, aes(x = hf_score)) +
  geom_histogram(bins = 20, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of human freedom scores in all years", x = "Human freedom score", y = "Number of countries")

# Evolution of Freedom Scores over time: example for selected countries
selected_countries <- c("United States", "Canada", "Germany", "China", "Brazil")
selected_data <- subset(data, countries %in% selected_countries)
ggplot(selected_data, aes(x = year, y = hf_score, color = countries)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Evolution of Freedom Scores Over Time", x = "Year", y = "Human Freedom Score")

# Scatterplot: Relationship between Personal and Economic Freedom
ggplot(data, aes(x = pf_score, y = ef_score)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Correlation between Personal and Economic Freedom", x = "Personal Freedom Score", y = "Economic Freedom Score")

#Global correlation matrix
# Clean graph window
plot.new()
dev.off()
# Select numeric columns only
hfi_data_num <- data %>%
  select(where(is.numeric))
# Calculate correlation matrix for all numeric columns
matrix_correlation <- cor(hfi_data_num, use = "complete.obs")
# Re-run corrplot with reduced text size
corrplot(matrix_correlation, method = "color", type = "upper", order = "hclust", tl.col = "black", tl.srt = 45, tl.cex = 0.01)


############################################################ Data cleaning

# Replace missing data with country average for previous years

data_no_empty <- data %>%
  group_by(countries) %>%
  mutate(across(where(is.numeric), 
                ~ ifelse(is.na(.), mean(., na.rm = TRUE), .),
                .names = "imputed_{.col}")) %>%
  ungroup()


##Problem with divorce rate in Suriname in 2020
# Calculate the average of the pf_identity_divorce column, excluding missing values
mean_pf_identity_divorce <- mean(data_no_empty$pf_identity_divorce, na.rm = TRUE)
# Replace missing value for Suriname in 2020 in column pf_identity_divorce
data_no_empty <- data_no_empty %>%
  mutate(pf_identity_divorce = replace(pf_identity_divorce, countries == "Suriname" & year == 2020 & is.na(pf_identity_divorce), mean_pf_identity_divorce))


# Identify the most recent year
last_year <- max(data_no_empty$year, na.rm = TRUE)
# Filter the data to keep only that of the last year
data_last_year <- data_no_empty %>%
  filter(year == last_year)

###To keep:countries, hf_score, pf_religion_freedom, pf_ss_homicide, pf_assembly_freedom, ef_money_inflation, pf_identity_divorce,

columns_keep <- c("countries", "hf_score", "pf_religion_freedom", "pf_ss_homicide", "pf_assembly", "ef_money_inflation", "pf_identity_divorce")

# Select only the specified columns
data_last_year <- data_last_year[columns_keep]

# Displays the first lines of the new data set
head(data_last_year)

# Check for missing values
sapply(data_last_year, function(x) sum(is.na(x)))

############################################################ Graphs

### Histogram of human freedom scores
ggplot(data_last_year, aes(x = hf_score)) +
  geom_histogram(bins = 20, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of human freedom scores in 2020", x = "Human freedom score", y = "Number of countries")

### Correlation matrix
# Install and load corrplot package if required
install.packages("corrplot")
library(corrplot)
# Calculate correlation matrix
cor_mat <- cor(data_last_year[, sapply(data_last_year, is.numeric)])
# Display correlation matrix
corrplot(cor_mat, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)


############################################################ Models

library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(glmnet)
library(class)
library(randomForest)
library(gbm)
library(caret)

# Separation of data into explanatory and target variables
X <- data_last_year[, !(names(data_last_year) %in% 'hf_score')]
y <- data_last_year$hf_score

# Division into training and test sets
set.seed(42)
index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[index, ]
X_test <- X[-index, ]
y_train <- y[index]
y_test <- y[-index]

# Data standardization for LASSO and Ridge
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)
y_train_vector <- as.vector(y_train)

############ GLM

# Select only columns of interest for the model
data_model <- select(data_last_year, hf_score, pf_religion_freedom, pf_ss_homicide, pf_assembly, ef_money_inflation, pf_identity_divorce)

# Build the GLM model
glm_model <- glm(hf_score ~ ., data = data_model, family = 'gaussian')

# View Model Summary
summary(glm_model)

############# LASSO

# Prepare data for glmnet
X <- as.matrix(data_model[, -1]) # Exclude hf_score column for predictors
y <- data_model$hf_score

# Divide data into training and test sets
set.seed(42)
index <- sample(1:nrow(X), nrow(X)*0.7)
X_train <- X[index, ]
X_test <- X[-index, ]
y_train <- y[index]
y_test <- y[-index]

X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale"))

# Adjust the LASSO model
cv_lasso <- cv.glmnet(X_train_scaled, y_train, alpha = 1)

# Find the optimal lambda
best_lambda <- cv_lasso$lambda.min

# Train the LASSO model with the best lambda
lasso_model <- glmnet(X_train_scaled, y_train, alpha = 1, lambda = best_lambda)

# Display the model coefficients
coef(lasso_model)

######### Ridge

# Ridge Regression
cv_ridge <- cv.glmnet(X_train_scaled, y_train, alpha = 0) # Alpha=0 pour Ridge
best_lambda_ridge <- cv_ridge$lambda.min
ridge_model <- glmnet(X_train_scaled, y_train, alpha = 0, lambda = best_lambda_ridge)
# View Ridge Model Coefficients
coef(ridge_model)

######### kNN

# Test differents values of k
k_values <- c(1, 3, 5, 10, 20)
for (k in k_values) {
  knn_pred <- knn(train = X_train_scaled, test = X_test_scaled, cl = y_train, k = k)
  knn_rmse <- RMSE(as.numeric(knn_pred), y_test)
  cat("k =", k, ": RMSE =", knn_rmse, "\n")
}

#This test shows us that for values 1, 3, 5, 10 and 20, the RMSE is still very high and around 48

k <- 1
knn_pred <- knn(train = X_train_scaled, test = X_test_scaled, cl = y_train, k = k)
knn_pred

######### Random Forest

# Random Forest
rf_model <- randomForest(x = X_train, y = y_train)
rf_pred <- predict(rf_model, X_test)
rf_pred

######### Boosting

# Convert X_test to data frame
X_test_df <- as.data.frame(X_test)
# Boosting
set.seed(42)
boost_model <- gbm(hf_score ~ ., data = data_model, distribution = "gaussian", n.trees = 100, interaction.depth = 3, shrinkage = 0.1)


############################################################ Models evaluation

install.packages("caret")
library(caret)
library(glmnet)
library(class)
library(randomForest)
library(gbm)

###LASSO
lasso_pred <- predict(lasso_model, s = best_lambda, newx = X_test_scaled)
lasso_rmse <- RMSE(lasso_pred, y_test)
lasso_r2 <- R2(lasso_pred, y_test)

###GLM
X_test_df <- as.data.frame(X_test)
glm_pred <- predict(glm_model, newdata = X_test_df)
glm_rmse <- RMSE(glm_pred, y_test)
glm_r2 <- R2(glm_pred, y_test)

###Ridge regression
ridge_pred <- predict(ridge_model, s = best_lambda_ridge, newx = X_test_scaled)
ridge_rmse <- RMSE(ridge_pred, y_test)
ridge_r2 <- R2(ridge_pred, y_test)

###kNN
# Convert kNN predictions to numeric
knn_pred_numeric <- as.numeric(knn_pred)
knn_rmse <- RMSE(knn_pred_numeric, y_test)
knn_r2 <- R2(knn_pred_numeric, y_test)

##Random Forest
rf_pred <- predict(rf_model, X_test)
rf_rmse <- RMSE(rf_pred, y_test)
rf_r2 <- R2(rf_pred, y_test)
# Compute the importance of variables in the random forest
rf_importance <- importance(rf_model)
# Show the importance of variables
print(rf_importance)
# Variable Importance Chart
varImpPlot(rf_model)

##Boosting
# X_test must be a.frame data
X_test_df <- as.data.frame(X_test)
boost_pred <- predict(boost_model, n.trees = 100, newdata = X_test_df)
boost_rmse <- RMSE(boost_pred, y_test)
boost_r2 <- R2(boost_pred, y_test)
# Compute and display the importance of variables for the Boosting model
boost_importance <- summary(boost_model, n.trees = 100)
# Show the importance of variables
print(boost_importance)
# Create a dataframe for the graphs
importance_df <- data.frame(
  Variable = rownames(boost_importance),
  Importance = boost_importance$rel.inf
)
# Graphs of the importance of the variables
ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Importance of variables", x = "Variables", y = "Importance")


####Show global results

# Function to calculate the correct error rate for regression
calculate_error_rate <- function(predictions, actual, threshold = 0.1) {
  error <- abs(predictions - actual) / actual
  mean(error > threshold)
}


### Error rate for the models - Evaluation

# For Ridge
ridge_error_rate <- calculate_error_rate(ridge_pred, y_test)

# For the kNN model
knn_error_rate <- calculate_error_rate(knn_pred_numeric, y_test)

# For the random forest model
rf_error_rate <- calculate_error_rate(rf_pred, y_test)

# For the Boosting model
boost_error_rate <- calculate_error_rate(boost_pred, y_test)

# For the GLM model
glm_error_rate <- calculate_error_rate(glm_pred, y_test)

# For the LASSO model
lasso_error_rate <- calculate_error_rate(lasso_pred, y_test)

### Show global results

cat("Ridge RMSE:", ridge_rmse, "R²:", ridge_r2, "Error Rate:", ridge_error_rate, "\n")
cat("kNN RMSE:", knn_rmse, "R²:", knn_r2, "Error Rate:", knn_error_rate, "\n")
cat("Random Forest RMSE:", rf_rmse, "R²:", rf_r2, "Error Rate:", rf_error_rate, "\n")
cat("Boosting RMSE:", boost_rmse, "R²:", boost_r2, "Error Rate:", boost_error_rate, "\n")
cat("GLM RMSE:", glm_rmse, "R²:", glm_r2, "Error Rate:", glm_error_rate, "\n")
cat("LASSO RMSE:", lasso_rmse, "R²:", lasso_r2, "Error Rate:", lasso_error_rate, "\n")

#These results suggest that for future prediction tasks on similar data, the Boosting model might be most appropriate. However, the GLM, LASSO and Ridge models are also quite reliable for this data set.


## Graphs of the performance

performance_data <- data.frame(
  Model = c("Ridge", "Boosting", "GLM", "LASSO", "RF"),
  RMSE = c(ridge_rmse, boost_rmse, glm_rmse, lasso_rmse, rf_rmse),
  R2 = c(ridge_r2, boost_r2, glm_r2, lasso_r2, rf_r2),
  ErrorRate = c(ridge_error_rate, boost_error_rate, glm_error_rate, lasso_error_rate, rf_error_rate)
)

# Graph for RMSE
ggplot(performance_data, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "RMSE of models", x = "Models", y = "RMSE")

# Graph for R²
ggplot(performance_data, aes(x = Model, y = R2, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "R² of models", x = "Models", y = "R²")

# Graph for Error Rate
ggplot(performance_data, aes(x = Model, y = ErrorRate, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Error rate of models", x = "Models", y = "Error rate")
