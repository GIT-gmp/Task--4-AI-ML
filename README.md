# Task--4-AI-ML
Build a binary classifier using logistic regression. 
Data Loading and Initial Preprocessing
Dataset Selection: The code utilizes the "Breast Cancer Wisconsin (Diagnostic)" dataset, which is a widely used benchmark for binary classification. The dataset is expected to be in a CSV file named data.csv.
Robust File Loading: A try-except block is implemented to handle potential FileNotFoundError, ensuring the script provides a clear message if the dataset is not found.
Initial Data Overview:
df.head(): Displays the first few rows of the DataFrame, offering an immediate understanding of its structure.
df.info(): Provides a concise summary, including column names, non-null counts, and data types. This helps in identifying missing values and confirming data types.
df.describe(): Generates descriptive statistics for numerical columns, such as mean, standard deviation, and quartiles, giving insights into the data's distribution.
Irrelevant Column Removal: The 'id' column (a unique identifier) and 'Unnamed: 32' (often an empty column in this specific dataset) are dropped as they do not contribute to the predictive power of the model.
Target Variable Transformation: The diagnosis column, which is the target variable, contains categorical values ('M' for Malignant, 'B' for Benign). These are converted into numerical format: 'M' is mapped to 1 (representing the positive class) and 'B' to 0 (representing the negative class). This numerical representation is essential for machine learning algorithms.
Class Distribution Check: df['diagnosis'].value_counts() is used to inspect the distribution of the target classes. This step is important for identifying potential class imbalance, which might influence model performance and evaluation strategies.
2. Data Splitting and Feature Standardization
Feature and Target Separation:
X: Represents the feature matrix, containing all independent variables (columns other than diagnosis).
y: Represents the target vector, containing the diagnosis column.
Train-Test Split: The sklearn.model_selection.train_test_split function is employed to divide the dataset into training and testing subsets.
test_size=0.2: Allocates 20% of the data for testing and 80% for training.
random_state=42: Ensures that the data split is deterministic and reproducible, meaning the same training and testing sets will be generated every time the code is run.
stratify=y: This is a critical parameter for classification tasks. It ensures that the proportion of classes in both the training and testing sets is maintained, mirroring the original dataset's class distribution. This helps prevent issues arising from imbalanced classes in the splits.
Feature Standardization: StandardScaler from sklearn.preprocessing is used to standardize the numerical features.
scaler.fit_transform(X_train): The scaler learns the mean and standard deviation from the training data and then transforms (scales) the training data.
scaler.transform(X_test): The same scaler (with parameters learned from the training data) is then used to transform the test data. This prevents "data leakage," where information from the test set could inadvertently influence the training process. Standardization is beneficial for Logistic Regression as it can help the optimization algorithm converge faster and improve model performance by giving equal importance to features with different scales.
