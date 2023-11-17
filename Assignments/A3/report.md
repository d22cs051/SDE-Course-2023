<h2><center> 
<p align="center" width="100%">
    SDE Assignment 3 Report<br>
    Bikash Dutta (D22CS051)
</p>

</center></h2>

## Code Explaination

### 1. Downloading Dataset:
```python
# Check if the dataset file exists, if not, download it from Kaggle
import os
if not os.path.isfile("pima-indians-diabetes.csv"):
    # Downloading the dataset using gdown
    !wget "https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv/download?datasetVersionNumber=1" -O "pima-indians-diabetes.csv"
    print("Downloading Done!")
else:
    print("File Already exists!!!")
```

Explanation:
- This section checks if the dataset file "pima-indians-diabetes.csv" exists in the current working directory.
- If the file does not exist, it downloads the dataset from Kaggle using the `gdown` command.
- The Kaggle dataset URL is provided within the code.
- The downloaded file is saved with the name "pima-indians-diabetes.csv".
- A message is printed indicating whether the download was successful or if the file already exists.

### 2. Loading Dataset:
```python
# Import necessary libraries (numpy, pandas)
import numpy as np
import pandas as pd

# Read the dataset into a Pandas DataFrame
ds = pd.read_csv("./pima-indians-diabetes.csv", names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"])
ds
```

Explanation:
- The code imports the necessary libraries, `numpy` and `pandas`.
- It reads the dataset from the CSV file into a Pandas DataFrame named `ds`.
- The column names are explicitly provided as ["Pregnancies", "Glucose", ..., "Class"] using the `names` parameter.
- The dataset is displayed, showing the first few rows.

### 3. Splitting Dataset:
```python
# Import train_test_split from sklearn
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_ds, y_ds, test_size=0.3, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
```

Explanation:
- The code imports the `train_test_split` function from `sklearn.model_selection`.
- It splits the dataset into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using a 70-30 ratio.
- The `random_state` parameter is set to ensure reproducibility of the split.
- The shapes of the training and testing sets are printed for verification.

### 4. Training Models:

#### a. Model Evaluation Function:
```python
# Install necessary libraries (matplotlib, seaborn)
%pip install matplotlib seaborn -q

# Import evaluation metrics and visualization libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score, log_loss, hamming_loss, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to evaluate and plot results
def evaluate_model(model, X_test, y_test):
    # ... (detailed explanation provided in the main report) ...
```

Explanation:
- This section installs the necessary libraries `matplotlib` and `seaborn` using the `%pip install` magic command.
- It imports various evaluation metrics and visualization libraries from `sklearn.metrics`, `matplotlib`, and `seaborn`.
- A function named `evaluate_model` is defined to calculate evaluation metrics, plot ROC curves, and Confusion Matrices.

#### b. Model 1: Logistic Regression
```python
# Scaling x and y for Logistic Regression
from sklearn.preprocessing import StandardScaler

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Train the Logistic Regression model
clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train_scaled, y_train)

# Output predicted probabilities and class for a specific example
clf.predict_proba([X_train_scaled[10]]), clf.predict([X_train[10]])

# Evaluate and plot metrics using the evaluate_model function
evaluate_model(clf, X_test=X_test_scaled, y_test=y_test)
```

Explanation:
- The input features (`X_train` and `X_test`) are scaled using `StandardScaler`.
- The Logistic Regression model is imported from `sklearn.linear_model` and trained on the scaled data.
- The trained model is used to predict probabilities and class for a specific example.
- The `evaluate_model` function is called to assess and visualize the model's performance.

#### c. Model 2: Decision Trees (Similar structure as Model 1)
```python
# ... (similar structure as Model 1, with DecisionTreeClassifier) ...
```

#### d. Model 3: Random Forest (Similar structure as Model 1)
```python
# ... (similar structure as Model 1, with RandomForestClassifier) ...
```

#### e. Model 4: Support Vector Machine (SVM) (Similar structure as Model 1)
```python
# ... (similar structure as Model 1, with SVC) ...
```

#### f. Model 5: Neural Networks (Similar structure as Model 1)
```python
# ... (similar structure as Model 1, with MLPClassifier) ...
```

Explanation (for Models 2-5):
- The input features are scaled using `StandardScaler`.
- The respective model is imported from `sklearn` and trained on the scaled data.
- Predictions, evaluation, and visualization are done using the `evaluate_model` function.

## Method Explaination
### 1. **Logistic Regression:**
   - **Working Principle:**
     - Logistic Regression is a linear classification algorithm commonly used for binary classification tasks.
     - It models the probability that an instance belongs to a particular class.
     - The algorithm applies the logistic function (sigmoid) to a linear combination of input features, mapping the result to a probability between 0 and 1.
     - A threshold is then applied to classify instances into one of two classes based on the calculated probabilities.

### 2. **Decision Trees:**
   - **Working Principle:**
     - Decision Trees are non-linear models that make decisions based on a series of rules.
     - The algorithm recursively splits the dataset based on features to create a tree-like structure.
     - At each node, the algorithm selects the feature that provides the best split, often using metrics like Gini impurity or information gain.
     - The process continues until a stopping criterion is met, creating leaf nodes that represent class labels.

### 3. **Random Forest:**
   - **Working Principle:**
     - Random Forest is an ensemble learning method that combines multiple Decision Trees to improve predictive performance and control overfitting.
     - It builds a collection of Decision Trees, each trained on a random subset of the training data and features.
     - During prediction, each tree "votes," and the class with the most votes is selected as the final prediction.
     - Random Forest provides better generalization and robustness compared to individual Decision Trees.

### 4. **Support Vector Machine (SVM):**
   - **Working Principle:**
     - Support Vector Machine is a powerful classification algorithm that finds a hyperplane to separate data points into different classes.
     - The algorithm aims to maximize the margin between classes, which is the distance between the hyperplane and the nearest data points from each class.
     - SVM is effective in high-dimensional spaces and can handle non-linear relationships through the use of kernel functions.
     - It works well for both binary and multiclass classification tasks.

### 5. **Neural Networks (MLP - Multi-Layer Perceptron):**
   - **Working Principle:**
     - Neural Networks are a class of machine learning models inspired by the structure and function of the human brain.
     - The Multi-Layer Perceptron (MLP) is a type of feedforward neural network with an input layer, one or more hidden layers, and an output layer.
     - Neurons in each layer are connected with weights, and non-linear activation functions introduce non-linearity to the model.
     - Backpropagation is used for training, adjusting the weights to minimize the difference between predicted and actual outputs.

These algorithms cover a range of techniques suitable for different types of datasets and classification tasks. The provided code evaluates their performance on the Pima Indians Diabetes dataset using various metrics and visualizations.

## Metrics Explaination

### 1. **Accuracy:**
   - **Explanation:**
     - Accuracy measures the proportion of correctly classified instances out of the total instances.
   - **Formulation:**
     $$ \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} $$

### 2. **Precision:**
   - **Explanation:**
     - Precision is the ratio of correctly predicted positive observations to the total predicted positives.
     - It is a measure of the model's ability to avoid false positives.
   - **Formulation:**
     $$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}} $$

### 3. **Recall (Sensitivity or True Positive Rate):**
   - **Explanation:**
     - Recall is the ratio of correctly predicted positive observations to the actual positives in the dataset.
     - It is a measure of the model's ability to identify all relevant instances.
   - **Formulation:**
     $$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}} $$

### 4. **F1 Score:**
   - **Explanation:**
     - F1 Score is the harmonic mean of precision and recall. It provides a balance between precision and recall.
   - **Formulation:**
     $$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}} $$

### 5. **ROC Curve (Receiver Operating Characteristic Curve):**
   - **Explanation:**
     - The ROC curve visualizes the performance of a binary classification model across different threshold settings.
     - It plots the True Positive Rate (Recall) against the False Positive Rate.
   - **Formulation:**
     - ROC curve is a graphical representation; no specific formula.

### 6. **ROC AUC (Area Under the ROC Curve):**
   - **Explanation:**
     - ROC AUC quantifies the area under the ROC curve, providing a single value to measure the model's discriminatory power.
   - **Formulation:**
     - Area under the ROC curve; a higher value indicates better model performance.

### 7. **Confusion Matrix:**
   - **Explanation:**
     - A confusion matrix shows the distribution of predicted and actual class labels, highlighting true positives, true negatives, false positives, and false negatives.
   - **Formulation:**
     $$
        \begin{bmatrix}
        TN & FP \\
        FN & TP
        \end{bmatrix}
    $$


### 8. **AUC-PR (Area Under the Precision-Recall Curve):**
   - **Explanation:**
     - Similar to ROC AUC, AUC-PR quantifies the area under the precision-recall curve, providing a summary measure of model performance.
   - **Formulation:**
     - Area under the precision-recall curve; higher values indicate better model performance.

### 9. **Matthews Correlation Coefficient (MCC):**
   - **Explanation:**
     - MCC is a correlation coefficient between the observed and predicted binary classifications.
     - It is particularly useful for imbalanced datasets.
   - **Formulation:**
     $$ \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} $$

### 10. **Balanced Accuracy:**
   - **Explanation:**
     - Balanced Accuracy calculates the arithmetic mean of sensitivity (recall) and specificity.
     - It provides a balanced measure for imbalanced datasets.
   - **Formulation:**
     $$ \text{Balanced Accuracy} = \frac{\text{Sensitivity + Specificity}}{2} $$

### 11. **Gini Coefficient (Gini Index):**
   - **Explanation:**
     - Gini Coefficient measures the inequality in a distribution. In classification, it quantifies the impurity of a set of class labels.
   - **Formulation:**
     $$ \text{Gini Index} = 1 - \sum_{i=1}^{C} p_i^2 $$
     where \( C \) is the number of classes, and \( p_i \) is the proportion of samples belonging to class \( i \).

### 12. **Cohen's Kappa:**
   - **Explanation:**
     - Cohen's Kappa measures the agreement between observed and predicted classifications, considering the possibility of random agreement.
   - **Formulation:**
     $$ \text{Cohen's Kappa} = \frac{\text{Observed Agreement} - \text{Expected Agreement}}{1 - \text{Expected Agreement}} $$

### 13. **Log Loss (Logarithmic Loss):**
   - **Explanation:**
     - Log Loss quantifies the accuracy of a classifier by penalizing false classifications.
     - Lower log loss values indicate better model performance.
   - **Formulation:**
     $$ \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$
     where \( N \) is the number of instances, \( y_i \) is the true class label (0 or 1), and \( p_i \) is the predicted probability.

### 14. **Hamming Loss:**
   - **Explanation:**
     - Hamming Loss measures the proportion of incorrectly predicted labels.
     - It is applicable to multi-label classification problems.
   - **Formulation:**
     $$ \text{Hamming Loss} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M} \sum_{j=1}^{M} I(y_{ij} \neq \hat{y}_{ij}) $$
     where \( N \) is the number of instances, \( M \) is the number of labels, \( y_{ij} \) is the true label, \( \hat{y}_{ij} \) is the predicted label, and \( I() \) is the indicator function.

### 15. **Jaccard Similarity Coefficient (Jaccard Index):**


   - **Explanation:**
     - Jaccard Index measures the similarity between two sets by comparing the intersection and union of their elements.
     - In classification, it quantifies the similarity between predicted and true class labels.
   - **Formulation:**
     $$ \text{Jaccard Index} = \frac{\text{Intersection of Sets}}{\text{Union of Sets}} $$

These evaluation metrics provide a comprehensive assessment of classification model performance from different perspectives. The choice of metrics depends on the specific goals and characteristics of the classification task.

## Results

### Model 1: Logistic Regression

**1. Numerical Results:**
   - *Numerical Results:* ![Numerical Results - Logistic Regression](https://i.imgur.com/hyfvASD.png)

**2. Graphical Results:**
   - *RoC Curve:* ![RoC - Logistic Regression](https://i.imgur.com/cN1yC1h.png)
   - *Confusion Matrix:* ![Confusion Matrix - Logistic Regression](https://i.imgur.com/Z28DeQq.png)

---

### Model 2: Decision Trees

**1. Numerical Results:**
   - *Numerical Results:* ![Numerical Results - Decision Trees](https://i.imgur.com/HVzJIlm.png)

**2. Graphical Results:**
   - *RoC Curve:* ![RoC - Decision Trees](https://i.imgur.com/we99oyv.png)
   - *Confusion Matrix:* ![Confusion Matrix - Decision Trees](https://i.imgur.com/NtRxbWL.png)


---

### Model 3: Random Forest

**1. Numerical Results:**
   - *Numerical Results:* ![Numerical Results - Random Forest](https://i.imgur.com/v7nK7re.png)

**2. Graphical Results:**
   - *RoC Curve:* ![RoC - Random Forest](https://i.imgur.com/HnjMO8B.png)
   - *Confusion Matrix:* ![Confusion Matrix - Random Forest](https://i.imgur.com/hgCi54m.png)

---

### Model 4: Support Vector Machine (SVM)

**1. Numerical Results:**
   - *Numerical Results:* ![Numerical Results - SVM](https://i.imgur.com/MApMEYz.png)

**2. Graphical Results:**
   - *RoC Curve:* ![RoC - SVM](https://i.imgur.com/SLZtrvV.png)
   - *Confusion Matrix:* ![Confusion Matrix - SVM](https://i.imgur.com/5ajvR9G.png)

---

### Model 5: Neural Networks

**1. Numerical Results:**
   - *Numerical Results:* ![Numerical Results - Neural Networks](https://i.imgur.com/IFV7ZS6.png)

**2. Graphical Results:**
   - *RoC Curve:* ![RoC - Neural Networks](https://i.imgur.com/awLtr40.png)
   - *Confusion Matrix:* ![Confusion Matrix - Neural Networks](https://i.imgur.com/BRGqZEi.png)

---

### Table for all the models

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC | AUC-PR | MCC   | Balanced Accuracy | Gini Coefficient | Cohen's Kappa | Log Loss | Hamming Loss | Jaccard Similarity Coefficient |
|----------------------|----------|-----------|--------|----------|---------|--------|-------|-------------------|------------------|---------------|----------|--------------|---------------------------------|
| Logistic Regression  | 0.7359   | 0.6173    | 0.6250 | 0.6211   | 0.7978  | 0.6635 | 0.4185| 0.7099            | 0.5957           | 0.4185        | 0.5228   | 0.2641       | 0.4505                          |
| Decision Trees       | 0.6840   | 0.5361    | 0.6500 | 0.5876   | 0.6760  | 0.6536 | 0.3393| 0.6760            | 0.3520           | 0.3352        | 11.3904  | 0.3160       | 0.4160                          |
| Random Forest        | 0.7576   | 0.6538    | 0.6375 | 0.6456   | 0.8056  | 0.6756 | 0.4615| 0.7293            | 0.6112           | 0.4614        | 0.5046   | 0.2424       | 0.4766                          |
| Support Vector Machine (SVM) | 0.7446   | 0.6438    | 0.5875 | 0.6144   | 0.7974  | 0.6464 | 0.4250| 0.7077            | 0.5947           | 0.4240        | 0.5220   | 0.2554       | 0.4434                          |
| Neural Networks      | 0.7013   | 0.5663    | 0.5875 | 0.5767   | 0.7293  | 0.6034 | 0.3462| 0.6745            | 0.4586           | 0.3460        | 1.9141   | 0.2987       | 0.4052                          |


---

## Conclusion

In this analysis, we performed a series of tasks related to a diabetes prediction dataset. The process included downloading the dataset, loading and splitting it, training multiple machine learning models, and evaluating their performance using various metrics. The models considered were Logistic Regression, Decision Trees, Random Forest, Support Vector Machine (SVM), and Neural Networks.

### Dataset Processing
- We began by downloading the dataset from Kaggle, checking for its existence, and loading it into a Pandas DataFrame. The dataset consists of features related to diabetes, such as pregnancies, glucose levels, blood pressure, and more.

### Model Training and Evaluation
1. **Logistic Regression:**
   - Achieved an accuracy of 73.59%.
   - Notable precision and recall values.
   - ROC AUC of 79.78% indicates good discriminatory power.
   - Additional metrics like AUC-PR, MCC, and balanced accuracy provide a comprehensive evaluation.

2. **Decision Trees:**
   - Accuracy of 68.40%.
   - Balanced precision and recall.
   - Notable AUC-PR, MCC, and balanced accuracy.
   - Log Loss and Hamming Loss metrics provide additional insights.

3. **Random Forest:**
   - Demonstrated a solid accuracy of 75.76%.
   - Balanced precision and recall values.
   - ROC AUC of 80.56% indicates strong discriminatory ability.
   - Additional metrics showcase the model's performance comprehensively.

4. **Support Vector Machine (SVM):**
   - Accuracy of 74.46%.
   - Balanced precision and recall.
   - Notable AUC-PR, MCC, and balanced accuracy.
   - Gini Coefficient and Cohen's Kappa provide additional evaluation.

5. **Neural Networks:**
   - Achieved an accuracy of 70.13%.
   - Balanced precision and recall.
   - ROC AUC of 72.93% indicates reasonable discriminatory power.
   - Various additional metrics contribute to a thorough evaluation.

### Overall Insights
- Each model exhibits strengths in different aspects, and the choice of the best model depends on the specific requirements of the problem at hand.
- Further tuning and optimization of hyperparameters may enhance the performance of these models.
- Visualization of ROC curves, confusion matrices, and other metrics aids in understanding the models' behavior.

This analysis provides a comprehensive overview of the dataset, model training, and evaluation. Further exploration and refinement of models can be conducted based on the specific goals and constraints of the application.