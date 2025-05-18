# Customer Churn Analysis Project

## Overview

This project analyzes customer churn using the "Telco Customer Churn" dataset. It is divided into two main parts:

1.  **Exploratory Data Analysis (EDA) and Preprocessing**:  This stage focuses on understanding the dataset, cleaning it, and preparing it for machine learning.
2.  **Model Development and Evaluation**:  This stage involves building several classification models to predict customer churn and evaluating their performance.

## Project Files

* **Churn Analysis Project.ipynb**:  This notebook covers the EDA and preprocessing steps and compares different models.
* **Model for Churn Analysis.ipynb**:  This notebook covers the model building and evaluation steps.
* **Telco Customer Churn.csv**: The dataset used for the analysis.

--

## 1. Churn Analysis Project.ipynb: EDA and Preprocessing

### 1.1.  Libraries Used

* `numpy`: For numerical operations.
* `pandas`: For data manipulation and analysis.
* `matplotlib.pyplot`: For creating plots.
* `seaborn`: For enhanced data visualization.
* `sklearn.preprocessing`: For scaling, encoding, and other preprocessing techniques.
* `sklearn.feature_selection`: For feature selection.
* `sklearn.ensemble`: For the RandomForestClassifier model.
* `sklearn.linear_model`: For the LogisticRegression model.
* `sklearn.svm`: For the Support Vector Classifier (SVC) model.
* `sklearn.decomposition`: For Principal Component Analysis (PCA).
* `sklearn.model_selection`: For K-Fold cross-validation.
* `sklearn.cluster`: For KMeans clustering.
* `sklearn.metrics`: For Adjusted Rand Index (ARI).

### 1.2. Data Loading and Initial Inspection

* Loads the "Telco Customer Churn.csv" dataset into a pandas DataFrame.
* Displays the first few rows using `Dataset.head()` to get a quick overview.
* Checks data types of columns using `Dataset.dtypes`.
* Identifies missing values using `Dataset.isnull().sum()`.
* Examines the shape of the dataset using `Dataset.shape`.

### 1.3. Feature Selection and Data Preparation

* Defines lists of features: `Features` (categorical features) and `Features2` (numerical features).
* Separates features (X) and target variable (y - "Churn").
* Converts numerical features in `Features2` to numeric, coercing errors (handling potential non-numeric values in 'TotalCharges').
* One-hot encodes categorical features in `Features` using `pd.get_dummies()`.
* Concatenates the encoded categorical features with the numerical features.
* Handles missing values in 'TotalCharges' by filling them with the mean.
* Scales numerical features using `StandardScaler`.
* Encodes the target variable "Churn" using `LabelEncoder`.

### 1.4. Feature Selection using Recursive Feature Elimination (RFE)

* Applies Recursive Feature Elimination with Logistic Regression (`RFE(LogisticRegression(), 10)`) to select the top 10 features.
* Prints the selected features.

### 1.5. Principal Component Analysis (PCA)

* Applies PCA to reduce the dimensionality of the feature space, retaining 90% of the variance.
* Transforms the data using the fitted PCA.

### 1.6. KMeans Clustering

* Applies KMeans clustering to the PCA-transformed data with 3 clusters.
* Calculates the Adjusted Rand Index (ARI) to compare the KMeans clusters with the actual churn labels.
* Prints the ARI score.
* Creates a DataFrame to compare the clusters with actual churn.
* Prints a table showing the distribution of churn within each cluster.

### 1.7. Key Findings from EDA

* The dataset contains a mix of categorical and numerical features.
* The "TotalCharges" column had some non-numeric values, which were handled by converting to numeric and filling missing values with the mean.
* PCA was used to reduce the dimensionality of the data.
* KMeans clustering did not effectively separate customers based on churn, as indicated by a low ARI score.

## 2. Model for Churn Analysis.ipynb: Model Development and Evaluation

### 2.1. Libraries Used

* `numpy`: For numerical operations.
* `pandas`: For data manipulation.
* `matplotlib.pyplot`: For plotting.
* `seaborn`: For data visualization.
* `sklearn.preprocessing`: For scaling and encoding.
* `sklearn.feature_selection`: For feature selection.
* `sklearn.ensemble`: For RandomForestClassifier.
* `sklearn.linear_model`: For LogisticRegression.
* `sklearn.svm`: For SVC.
* `sklearn.decomposition`: For PCA.
* `sklearn.model_selection`: For KFold cross-validation and train-test split.
* `sklearn.metrics`: For various evaluation metrics (ROC curve, AUC, confusion matrix, accuracy, recall, precision, F1-score, ROC AUC score).

### 2.2. Data Loading and Preprocessing

* Loads the "Telco Customer Churn.csv" dataset.
* Performs similar preprocessing steps as in "Churn Analysis Project.ipynb":
    * Identifies binary columns.
    * Handles imbalanced classes (if present - though not explicitly detailed in provided snippets).
    * Encodes categorical variables.
    * Converts 'TotalCharges' to numeric.
    * Scales numerical features.
    * Encodes the target variable "Churn".

### 2.3. Model Training and Evaluation

* Splits the data into training and testing sets.
* **Logistic Regression:**
    * Trains a Logistic Regression model.
    * Evaluates the model using metrics like accuracy, recall, precision, F1-score, and ROC AUC score.
    * Plots the ROC curve and calculates AUC.
    * Displays the confusion matrix.
    * Performs cross-validation.
* **Random Forest Classifier:**
    * Trains a Random Forest Classifier model.
    * Evaluates the model using the same metrics as Logistic Regression.
    * Plots the ROC curve and calculates AUC.
    * Displays the confusion matrix.
    * Performs cross-validation.
* **Support Vector Classifier (SVC):**
    * Trains an SVC model.
    * Evaluates the model using the same metrics.
    * Plots the ROC curve and calculates AUC.
    * Displays the confusion matrix.
    * Performs cross-validation.

### 2.4. Model Comparison and Selection

* Compares the performance of all three models based on the evaluation metrics.
* Selects the best-performing model (based on metrics like AUC, F1-score, etc.).

### 2.5. Results and Conclusion

* Summarizes the performance of the models.

## How to Run the Project

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
2.  **Place Data:** Ensure "Telco Customer Churn.csv" is in the same directory as the notebooks.
3.  **Run Notebooks:** Open and execute "Churn Analysis Project.ipynb" and "Model for Churn Analysis.ipynb" in a Jupyter environment.

##  Important Notes

* The notebooks contain detailed code, outputs, and visualizations for each step.
* The specific model performance metrics and conclusions may vary slightly depending on the exact execution and any randomness in the algorithms.
* Further hyperparameter tuning and model optimization could potentially improve the results.
* The insights from this analysis can inform customer retention strategies and help identify customers at high risk of churn.
