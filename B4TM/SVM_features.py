import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
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

# Set up the SVM classifier with a linear kernel
svm = SVC(kernel='linear')

# Perform recursive feature elimination (RFE) to select top 10 features
selector = RFE(svm, n_features_to_select=10, importance_getter='auto')
selector = selector.fit(X_train, y_train)

# Get the selected features
selected_features = [
    1160, 2218, 2379, 2263, 523, 495, 613, 1483, 1587, 1448,
    1527, 2067, 2108, 2184, 2794, 2742, 999, 1097, 318, 1306,
    274, 914, 949, 716, 2020, 1365, 2411, 1191, 2448, 1770,
    2723, 67, 231, 464, 2697, 754, 1929, 1280, 1872, 1788,
    1625, 2515, 618, 63, 2547, 2646, 672, 416, 1952, 1634
]

# Save the selected features to a file
with open("selected_features_SVM.txt", "w") as f:
    for feature_index in selected_features:
        f.write(f"{feature_index}\n")

# Train the SVM classifier using the selected features
svm.fit(X_train.iloc[:, selected_features], y_train)

# Predict on the validation set
y_pred_val = svm.predict(X_val.iloc[:, selected_features])

# Calculate accuracy for the validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)
