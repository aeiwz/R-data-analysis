# ============================
# R DATA ANALYSIS PIPELINE EXAMPLE
# Using BreastCancer dataset from mlbench
# ============================

# Load libraries
library(mlbench)       # Dataset
library(dplyr)
library(tidyr)
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(caret)      # For model training
library(cluster)    # Clustering algorithms
library(purrr)      # Functional programming
library(randomForest)

# Step 1: Load data
data("BreastCancer")
df <- BreastCancer

# Step 2: Initial exploration
glimpse(df)
summary(df)
head(df)

# Step 3: Data cleaning and filtering
# Remove ID column
df_clean <- df %>%
  select(-Id) %>%
  filter(!is.na(Class))

# Convert factor columns (except Class) to numeric
df_clean <- df_clean %>%
  mutate(across(where(is.factor) & !matches("Class"),
                ~ as.numeric(as.character(.))))

# Step 4: Handle missing data
# Option A: Remove rows with missing values
df_no_na <- df_clean %>%
  drop_na()

# Option B: Impute missing values (mean for numeric)
impute_mean <- function(x) ifelse(is.na(x), mean(x, na.rm=TRUE), x)
impute_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

df_imputed <- df_clean %>%
  mutate(across(where(is.numeric), impute_mean)) %>%
  mutate(across(where(is.character), ~replace_na(., impute_mode(.))))

# Step 5: Feature engineering (optional)
# No extra features here

# Step 6: Data normalization/scaling
numeric_cols <- df_imputed %>%
  select(where(is.numeric)) %>%
  names()

df_scaled <- df_imputed
df_scaled[numeric_cols] <- scale(df_imputed[numeric_cols])

# Step 7: Exploratory Data Analysis (EDA)
# Histogram of target variable
ggplot(df_scaled, aes(x = Class)) + geom_bar() + ggtitle("Distribution of Diagnosis Class")

# Correlation heatmap of numeric variables
cor_matrix <- cor(df_scaled[numeric_cols], use = "pairwise.complete.obs")
heatmap(cor_matrix, main = "Correlation Matrix")

# Step 8: Statistical testing
# t-test on Cell.size between classes
t_test_res <- t.test(Cell.size ~ Class, data = df_scaled)
print(t_test_res)

# Step 9: Multivariate Analysis

# PCA
pca_res <- PCA(df_scaled[numeric_cols], graph = FALSE)
fviz_pca_ind(pca_res, geom.ind = "point", col.ind = df_scaled$Class, addEllipses = TRUE)

# Clustering (k-means)
set.seed(123)
k <- 2
km_res <- kmeans(df_scaled[numeric_cols], centers = k, nstart = 25)

df_scaled$cluster <- factor(km_res$cluster)

fviz_cluster(km_res, data = df_scaled[numeric_cols])

# Step 10: (Optional) Model building (classification example)

set.seed(123)
trainIndex <- createDataPartition(df_scaled$Class, p = .8, list = FALSE)
trainData <- df_scaled[trainIndex, ]
testData <- df_scaled[-trainIndex, ]

rf_model <- randomForest(Class ~ ., data = trainData, importance = TRUE)
predictions <- predict(rf_model, testData)

conf_mat <- confusionMatrix(predictions, testData$Class)
print(conf_mat)

# ===============================
# END OF PIPELINE
# ===============================
