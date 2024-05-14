import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

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

def objective(trial):
    # Set up the SVM classifier with hyperparameters to optimize
    svm = SVC(C=trial.suggest_loguniform('C', 1e-2, 1e+2),
              gamma=trial.suggest_loguniform('gamma', 1e-6, 1e+1),
              kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']))

    # Train the SVM classifier
    svm.fit(X_train, y_train)

    # Predict on the validation set
    y_pred_validation = svm.predict(X_val)

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

# Train the SVM classifier with the best hyperparameters
best_svm = SVC(C=best_C, gamma=best_gamma, kernel=best_kernel)
best_svm.fit(X_train, y_train)

# Initialize lists to store number of features and corresponding accuracies
num_features_list = []
accuracy_list = []

# Iterate over different number of features to find the one with maximum accuracy
for k in range(1, X_train.shape[1] + 1):
    # Perform feature selection
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)

    # Train the SVM classifier
    best_svm.fit(X_train_selected, y_train)

    # Predict on the validation set
    y_pred_validation = best_svm.predict(X_val_selected)

    # Calculate accuracy for validation set
    accuracy_validation = accuracy_score(y_val, y_pred_validation)

    # Append current number of features and accuracy to the lists
    num_features_list.append(k)
    accuracy_list.append(accuracy_validation)

# Find the index of the maximum accuracy
max_accuracy_index = accuracy_list.index(max(accuracy_list))

# Plot number of features against accuracy
plt.plot(num_features_list, accuracy_list)
plt.scatter(num_features_list[max_accuracy_index], accuracy_list[max_accuracy_index], color='red', label='Max Accuracy')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Number of Features vs Accuracy')
plt.legend()
plt.show()

# Print the optimal number of features
print("Optimal Number of Features:", num_features_list[max_accuracy_index])
