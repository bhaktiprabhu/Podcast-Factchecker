"""This module includes functions for performance evaluation"""

import os
from typing import List

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def print_confusion_matrix(true_labels: List, predicted_labels: List):
    """Displays Confusion Matrix for model evaluation

    Args:
        true_labels (List): True values
        predicted_labels (List): Predicted Values
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(conf_matrix)


def print_classification_report(true_labels: List, predicted_labels: List):
    """Displays Classification Report for model evaluation

    Args:
        true_labels (List): True values
        predicted_labels (List): Predicted Values
    """
    report = classification_report(true_labels, predicted_labels)
    print(report)


def model_performance_evaluation(model_name: str):
    """Evaluate model performance for CSV files in a folder
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "output")
    true_labels = []
    predicted_labels = []
    
    true_column = "majority_claim"
    
    if model_name == 'factiverse':
        predicted_column = "factiverse_claim"
    elif model_name == 'openai':
        predicted_column = "openai_gpt4_claim"
    else:
        print('❌Unknown Model❌')
        return

    for file_name in os.listdir(folder_path):
        if file_name.endswith("_claim_analysis.csv"):
            file_path = os.path.join(folder_path, file_name)

            df = pd.read_csv(file_path)
            df = df.dropna(subset=['majority_claim'])

            if true_column in df.columns and predicted_column in df.columns:
                true_labels.extend(df[true_column].tolist())
                predicted_labels.extend(df[predicted_column].tolist())

    # Evaluate model performance using extracted labels
    print('***** Model Performance Evaluation for ' + model_name + '*****')
    print_confusion_matrix(true_labels, predicted_labels)
    print_classification_report(true_labels, predicted_labels)
