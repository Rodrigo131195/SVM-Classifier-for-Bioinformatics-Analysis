import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

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

# Ensure class labels in y_val are present in the index
y_val = y_val.reset_index(drop=True)

# Train the SVM classifier with OneVsRest strategy
svm = OneVsRestClassifier(SVC(kernel='linear', probability=True))
svm.fit(X_train, y_train)

# Predict probabilities for the validation set
y_probs = svm.decision_function(X_val)

# Binarize the labels
y_val_bin = label_binarize(y_val, classes=np.unique(y_val))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_val_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
for i in range(y_val_bin.shape[1]):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve (class {0}) (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")
plt.show()

