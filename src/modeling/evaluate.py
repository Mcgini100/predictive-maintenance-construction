# -*- coding: utf-8 -*-
"""
Functions for evaluating model performance.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any
import numpy as np

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculates standard classification metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class.
                                      Required for ROC AUC. Defaults to None.

    Returns:
        Dict[str, float]: A dictionary containing metric names and their values.
    """
    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        if y_prob is not None:
            try:
                # Ensure y_prob corresponds to the positive class (class 1)
                # If y_prob has two columns, take the second one.
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                     positive_class_prob = y_prob[:, 1]
                else:
                     positive_class_prob = y_prob # Assume it's already probability of positive class

                metrics['roc_auc'] = roc_auc_score(y_true, positive_class_prob)
            except ValueError as e:
                print(f"Warning: ROC AUC could not be calculated. Ensure y_true contains multiple classes. Error: {e}")
                metrics['roc_auc'] = np.nan # Or None, or 0.0 depending on preference
        else:
             metrics['roc_auc'] = np.nan # Indicate ROC AUC wasn't calculated

    except Exception as e:
        print(f"Error calculating metrics: {e}")

    return metrics

def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
    """
    Prints a comprehensive evaluation report including metrics and confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class.
    """
    print("\n--- Model Evaluation Report ---")

    # Calculate standard metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print("\nKey Metrics:")
    for name, value in metrics.items():
        print(f"  {name.capitalize()}: {value:.4f}")

    # Print Classification Report
    print("\nClassification Report:")
    try:
        # Get unique labels present in both y_true and y_pred for target_names
        labels = np.unique(np.concatenate((y_true, y_pred)))
        target_names = [f'Class {label}' for label in labels]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    try:
        cm = confusion_matrix(y_true, y_pred)
        print(pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                           columns=['Predicted Negative', 'Predicted Positive']))
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

    print("--- End of Evaluation Report ---")


# Example usage (can be commented out)
# if __name__ == '__main__':
#     # Dummy data
#     y_true_dummy = np.array([0, 1, 0, 1, 1, 0, 0, 1])
#     y_pred_dummy = np.array([0, 1, 1, 1, 0, 0, 0, 1])
#     y_prob_dummy = np.array([0.1, 0.8, 0.6, 0.9, 0.4, 0.2, 0.3, 0.7]) # Prob of class 1
#
#     print_evaluation_report(y_true_dummy, y_pred_dummy, y_prob_dummy)