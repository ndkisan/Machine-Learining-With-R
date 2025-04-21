# Load required libraries
library(tidyverse)
library(lubridate)
library(scales)
library(caret)
library(forcats)
library(ggplot2)
library(cluster)



# Load the dataset
data <- read.csv("C:\\Users\\RISVAN\\Downloads\\Warehouse_and_Retail_Sales.csv")

# Convert YEAR and MONTH into a proper Date column
data$Date <- make_date(data$YEAR, data$MONTH, 1)

# Replace missing values in sales columns with 0
data$RETAIL.SALES[is.na(data$RETAIL.SALES)] <- 0
data$RETAIL.TRANSFERS[is.na(data$RETAIL.TRANSFERS)] <- 0
data$WAREHOUSE.SALES[is.na(data$WAREHOUSE.SALES)] <- 0
print(data)


# 1. Retail Sales Over Time
monthly_sales <- data %>%
  group_by(Date) %>%
  summarise(Retail_Sales = sum(RETAIL.SALES),
            Warehouse_Sales = sum(WAREHOUSE.SALES))

ggplot(monthly_sales, aes(x = Date)) +
  geom_line(aes(y = Retail_Sales, color = "Retail Sales")) +
  geom_line(aes(y = Warehouse_Sales, color = "Warehouse Sales")) +
  labs(title = "Monthly Sales Trends (Retail vs Warehouse)",
       x = "Date", y = "Sales Volume", color = "Sales Type") +
  theme_minimal()

# 2. Total Sales by Item Type
item_type_sales <- data %>%
  group_by(ITEM.TYPE) %>%
  summarise(Total_Retail_Sales = sum(RETAIL.SALES)) %>%
  arrange(desc(Total_Retail_Sales))

ggplot(item_type_sales, aes(x = reorder(ITEM.TYPE, -Total_Retail_Sales), y = Total_Retail_Sales)) +
  geom_bar(stat = "identity", fill = "#377eb8") +
  labs(title = "Total Retail Sales by Item Type",
       x = "Item Type", y = "Total Retail Sales") +
  scale_y_continuous(labels = comma) +  
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Data Preparation
# Create binary classification target: High_Seller
median_sales <- median(data$RETAIL.SALES)
data$High_Seller <- ifelse(data$RETAIL.SALES > median_sales, 1, 0)

# Select features for modeling
model_data <- data %>%
  select(High_Seller, WAREHOUSE.SALES, RETAIL.TRANSFERS, MONTH, ITEM.TYPE) %>%
  drop_na()

# Handle categorical variable: ITEM.TYPE
model_data$ITEM.TYPE <- as.factor(model_data$ITEM.TYPE)
model_data$ITEM.TYPE <- fct_lump_n(model_data$ITEM.TYPE, n = 5)

# Make High_Seller a factor with valid names ("Yes", "No")
model_data$High_Seller <- factor(ifelse(model_data$High_Seller == 1, "Yes", "No"))

# Set seed for reproducibility
set.seed(123)

# Setup 10-fold cross-validation
train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Train logistic regression model using caret
logit_model <- train(
  High_Seller ~ .,
  data = model_data,
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "ROC"  # Evaluate based on ROC-AUC
)

# Print model results
print(logit_model)

# Confusion matrix based on cross-validated predictions
conf_matrix <- confusionMatrix(
  logit_model$pred$pred,
  logit_model$pred$obs
)
print(conf_matrix)





cv_df <- logit_model$resample

# If it has ROC scores instead of Accuracy:
ggplot(cv_df, aes(x = Resample, y = ROC)) +
  geom_col(fill = "#4c78a8") +
  labs(
    title = "Figure 1: Logistic Regression ROC (10-Fold Cross-Validation)",
    x = "Fold",
    y = "ROC Score"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Clustering with standardization
library(cluster)

cluster_data <- data %>%
  group_by(ITEM.CODE) %>%
  summarise(Avg_Warehouse = mean(WAREHOUSE.SALES),
            Avg_Transfers = mean(RETAIL.TRANSFERS)) %>%
  drop_na()

# Standardize
scaled_data <- scale(cluster_data[, -1])

# Elbow Method
wss <- sapply(1:10, function(k) {
  kmeans(scaled_data, centers = k, nstart = 10)$tot.withinss
})
plot(1:10, wss, type = "b", main = "Elbow Method for K", xlab = "Number of Clusters", ylab = "WSS")

# K-means with 3 clusters
set.seed(123)
kmeans_model <- kmeans(scaled_data, centers = 3, nstart = 25)
cluster_data$Cluster <- as.factor(kmeans_model$cluster)

#cluster plot
ggplot(cluster_data, aes(x = Avg_Warehouse, y = Avg_Transfers, color = Cluster)) +
  geom_point(size = 2) +
  labs(title = "K-Means Clustering of Product Behavior",
       x = "Average Warehouse Sales", y = "Average Retail Transfers") +
  theme_minimal()

