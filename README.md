# SVM-Classifier-for-Bioinformatics-Analysis

Description:
This project involves the implementation of a Support Vector Machine (SVM) classifier for bioinformatics data analysis. The SVM classifier is trained on a dataset containing features extracted from bioinformatics data, and it aims to predict the subgroup label for each sample. Additionally, feature selection and hyperparameter optimization techniques are explored to improve classifier performance.

Files:

SVM.py: This file contains the implementation of the SVM classifier.

Input Data Files:
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/train_call.csv
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/Train_call.csv
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/Train_call.txt
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/Train_clinical.csv
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/Train_clinical.txt
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/Validation_call.csv
/home/rodrigo/Documents/Bioinformatics_&_SB/B4TM/Validation_call.txt
Functionality:
Loads training data and labels.
Transposes the training data for further processing.
Merges dataframes based on sample indices.
Splits the data into training and validation sets.
Trains the SVM classifier using the One-vs-Rest strategy.
Predicts probabilities for the validation set.
Computes ROC curves and AUC scores for each class.
Plots ROC curves for visualization.
ROC_SVM.py: This file contains the code for computing ROC curves and AUC scores for the SVM classifier.

Functionality:
Imports necessary libraries.
Loads training data and labels.
Transposes the training data for further processing.
Merges dataframes based on sample indices.
Splits the data into training and validation sets.
Trains the SVM classifier using the One-vs-Rest strategy.
Predicts probabilities for the validation set.
Binarizes the labels.
Computes ROC curves and AUC scores for each class.
Plots ROC curves for visualization.
SVM_feature_number.py: This file explores feature selection and hyperparameter optimization for the SVM classifier.

Functionality:
Loads training data and labels.
Performs hyperparameter tuning using Optuna.
Finds the optimal number of features by iterating over different feature subsets.
Plots the number of features against accuracy to visualize feature selection.
Prints the optimal number of features.
SVM_features.py: This file implements feature selection for the SVM classifier.

Functionality:
Loads training data and labels.
Performs recursive feature elimination (RFE) to select top features.
Saves the selected features to a file.
Trains the SVM classifier using the selected features.
Prints the validation accuracy.
analysis.py: This file conducts data analysis using KMeans clustering.

Functionality:
Loads training data and target labels.
Encodes string labels to integer labels.
Tries different values of k for KMeans clustering.
Plots the sum of squared distances against the number of clusters to find the optimal value of k.
Usage:

Ensure that all input data files specified in the Python scripts are present in the specified locations.
Run the desired Python script(s) to perform various tasks such as training the SVM classifier, visualizing performance metrics, conducting feature selection, or analyzing data using KMeans clustering.
Dependencies:

pandas
matplotlib
scikit-learn
optuna
numpy
Note:

Each Python script serves a specific purpose within the project, ranging from classifier training to data analysis and visualization.
Users can explore different aspects of the project by running the corresponding scripts.
