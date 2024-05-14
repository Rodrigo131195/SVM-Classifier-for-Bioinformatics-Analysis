import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Load training data and labels
train_data = pd.read_csv("Train_call.csv")
train_labels = pd.read_csv("Train_clinical.csv")

# Transpose train_data to have samples as rows
train_labels = train_labels.iloc[1:, :]
train_data = train_data.iloc[:, 4:]
train_data_transposed = train_data.transpose()
train_data_transposed.reset_index(inplace=True)

# Merge the two DataFrames based on the index of 'train_data_transposed' and the 'Sample' column of 'train_labels'
merged_df_train = pd.merge(train_data_transposed, train_labels, left_on='index', right_on='Sample')

# Split the data into features (X) and target (y) for training data
X_train_val = merged_df_train.drop(['Sample', 'Subgroup', 'index'], axis=1)  # Features
y_train_val = merged_df_train['Subgroup']  # Target

# Split the data into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Define X_train_selected globally
X_train_selected = None

def objective(trial):
    global X_train_selected  # Declare as global to modify it within the function

    # Select hyperparameters to optimize
    C = trial.suggest_loguniform('C', 1e-2, 1e+2)
    gamma = trial.suggest_loguniform('gamma', 1e-6, 1e+1)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])

    # Select the top 41 features using F-score
    selector = SelectKBest(f_classif, k=41)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)

    # Train SVM classifier with selected hyperparameters
    svm = SVC(C=C, gamma=gamma, kernel=kernel)
    svm.fit(X_train_selected, y_train)

    # Predict on validation set
    y_pred_validation = svm.predict(X_val_selected)

    # Calculate accuracy for validation set
    accuracy_validation = accuracy_score(y_val, y_pred_validation)

    return accuracy_validation


# Perform hyperparameter tuning using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
best_C = best_params['C']
best_gamma = best_params['gamma']
best_kernel = best_params['kernel']

# Train SVM classifier with the best hyperparameters
best_svm = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
best_svm.fit(X_train_selected, y_train)

# Print the optimal hyperparameters
print("Optimal Hyperparameters (SVM):")
print("C:", best_C)
print("Gamma:", best_gamma)
print("Kernel:", best_kernel)

# Get selected features
selector = SelectKBest(f_classif, k=41)
X_train_selected = selector.fit_transform(X_train, y_train)
selected_features_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_features_indices]

# Print the optimal features
print("\nOptimal Features:")
for feature in selected_features:
    print(feature)
