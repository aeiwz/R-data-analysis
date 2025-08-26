# 🔬 R Data Analysis Pipeline Example  
**Dataset:** BreastCancer (from `mlbench`)  

This pipeline demonstrates a complete data science workflow in R, from **loading and cleaning data** to **exploration, statistical testing, multivariate analysis, clustering, and classification**.

---

## 📦 Step 1: Load Required Libraries
```r
library(mlbench)       # BreastCancer dataset
library(dplyr)         # Data manipulation
library(tidyr)         # Handling missing values
library(ggplot2)       # Data visualization
library(FactoMineR)    # Multivariate analysis (PCA)
library(factoextra)    # Visualizing PCA/Clustering
library(caret)         # Data partition & ML training
library(cluster)       # Clustering algorithms
library(purrr)         # Functional programming helpers
library(randomForest)  # Classification model
```

**Explanation:**  
We load all libraries needed for **data wrangling, visualization, multivariate analysis, clustering, and machine learning**.

---

## 📂 Step 2: Load Data
```r
data("BreastCancer")
df <- BreastCancer
```

**Explanation:**  
The dataset `BreastCancer` contains **699 breast cancer samples** with cytological characteristics and a target variable `Class` (benign/malignant).

---

## 🔎 Step 3: Initial Exploration
```r
glimpse(df)
summary(df)
head(df)
```

**Explanation:**  
- `glimpse()` gives an overview of data structure.  
- `summary()` provides statistical summaries.  
- `head()` shows the first rows for a quick check.  

---

## 🧹 Step 4: Data Cleaning
```r
# Remove ID column and ensure Class is valid
df_clean <- df %>%
  select(-Id) %>%
  filter(!is.na(Class))

# Convert factor columns (except Class) to numeric
df_clean <- df_clean %>%
  mutate(across(where(is.factor) & !matches("Class"),
                ~ as.numeric(as.character(.))))
```

**Explanation:**  
- Remove the **ID column** (not useful for modeling).  
- Convert factor columns like `Cl.thickness`, `Cell.size`, etc., into numeric values for analysis.  

---

## 🧩 Step 5: Handle Missing Data
Two approaches:  
### (A) Remove missing rows
```r
df_no_na <- df_clean %>% drop_na()
```

### (B) Impute missing values
```r
impute_mean <- function(x) ifelse(is.na(x), mean(x, na.rm=TRUE), x)
impute_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

df_imputed <- df_clean %>%
  mutate(across(where(is.numeric), impute_mean)) %>%
  mutate(across(where(is.character), ~replace_na(., impute_mode(.))))
```

**Explanation:**  
- Option A: Delete missing values (risk: lose data).  
- Option B: Replace missing numeric values with **mean**, categorical values with **mode**.  

---

## ⚖️ Step 6: Normalization
```r
numeric_cols <- df_imputed %>%
  select(where(is.numeric)) %>%
  names()

df_scaled <- df_imputed
df_scaled[numeric_cols] <- scale(df_imputed[numeric_cols])
```

**Explanation:**  
Scaling ensures all features are comparable, preventing large-value variables (like `Cl.thickness`) from dominating the analysis.

---

## 📊 Step 7: Exploratory Data Analysis (EDA)
### Distribution of classes
```r
ggplot(df_scaled, aes(x = Class)) +
  geom_bar() +
  ggtitle("Distribution of Diagnosis Class")
```

### Correlation heatmap
```r
cor_matrix <- cor(df_scaled[numeric_cols], use = "pairwise.complete.obs")
heatmap(cor_matrix, main = "Correlation Matrix")
```

---

## 📑 Step 8: Statistical Testing
```r
t_test_res <- t.test(Cell.size ~ Class, data = df_scaled)
print(t_test_res)
```

**Explanation:**  
A **t-test** compares whether the average `Cell.size` differs significantly between **benign** and **malignant** groups.

---

## 🔀 Step 9: Multivariate Analysis

### PCA (Principal Component Analysis)
```r
pca_res <- PCA(df_scaled[numeric_cols], graph = FALSE)
fviz_pca_ind(pca_res, geom.ind = "point", col.ind = df_scaled$Class, addEllipses = TRUE)
```

### K-means clustering
```r
set.seed(123)
k <- 2
km_res <- kmeans(df_scaled[numeric_cols], centers = k, nstart = 25)

df_scaled$cluster <- factor(km_res$cluster)

fviz_cluster(km_res, data = df_scaled[numeric_cols])
```

**Explanation:**  
- PCA reduces dimensionality while preserving variance → visualize cancer classes.  
- K-means clustering groups samples into **clusters** (here k=2 for benign vs malignant).  

---

## 🤖 Step 10: Model Building (Classification)

```r
set.seed(123)
trainIndex <- createDataPartition(df_scaled$Class, p = .8, list = FALSE)
trainData <- df_scaled[trainIndex, ]
testData <- df_scaled[-trainIndex, ]

rf_model <- randomForest(Class ~ ., data = trainData, importance = TRUE)
predictions <- predict(rf_model, testData)

conf_mat <- confusionMatrix(predictions, testData$Class)
print(conf_mat)
```

**Explanation:**  
- Use **Random Forest** for classification.  
- Train on 80% of data, test on 20%.  
- Evaluate using **confusion matrix** (accuracy, sensitivity, specificity).  

---

# ✅ Summary
This pipeline covers:  
- Data loading & cleaning  
- Handling missing values  
- Normalization & EDA  
- Statistical testing  
- Multivariate analysis (PCA & Clustering)  
- Classification (Random Forest)  

A full, reproducible workflow for **teaching and practicing data analysis in R**.  
