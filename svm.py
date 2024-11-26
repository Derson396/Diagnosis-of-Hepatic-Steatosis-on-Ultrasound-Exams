from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import re

# Load the CSV file
file_path = 'Rois/Planilhas/texture_data.csv'
data = pd.read_csv(file_path)

# Identify numeric columns (excluding the target column 'Classe')
feature_columns = data.columns[2:]


# Step 1: Remove outliers using IQR
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


data_cleaned = remove_outliers(data, feature_columns)

# Step 2: Apply robust scaling
scaler = RobustScaler()
data_scaled = data_cleaned.copy()
data_scaled[feature_columns] = scaler.fit_transform(data_cleaned[feature_columns])

# Step 3: Extract patient IDs from the 'Arquivo' column using regex
data_scaled['Patient_ID'] = data_scaled['Arquivo'].apply(lambda x: int(re.search(r'ROI_(\d+)_', x).group(1)))

# Step 4: Prepare for Leave-One-Group-Out Cross-Validation
logo = LeaveOneGroupOut()
X = data_scaled[feature_columns]
y = data_scaled['Classe']
groups = data_scaled['Patient_ID']

# Define the SVM hyperparameter grid
param_grid = {
    'C': [1, 100],
    'gamma': ['scale', 'auto', 1],
    'kernel': ['rbf']
}

# Placeholder for results
results = []

# Perform Leave-One-Group-Out Cross-Validation
for train_idx, test_idx in logo.split(X, y, groups):
    # Split into train and test sets for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Perform grid search on the training set with recall as the optimization metric
    grid_search = GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid,
        scoring='recall',  # Use recall as the optimization criterion during grid search
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate on the test set using accuracy
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Testando com paciente: {groups.iloc[test_idx].iloc[0]}")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Acurácia no teste: {accuracy:.2f}")

    # Store results
    results.append({
        'Test_Patient_ID': groups.iloc[test_idx].iloc[0],
        'Best_Params': grid_search.best_params_,
        'Test_Accuracy': accuracy,
        'Classification_Report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    })

# Summarize overall results
overall_accuracy = np.mean([result['Test_Accuracy'] for result in results])
overall_accuracy, len(results)  # Return overall accuracy and number of LOOCV folds

print(f'Overall Accuracy: {overall_accuracy:.2f}')
