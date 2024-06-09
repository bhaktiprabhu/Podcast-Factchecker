"""This module includes functions for performance evaluation of AI Tools"""

import os
from typing import List

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src import utils


def print_confusion_matrix(true_labels: List, predicted_labels: List):
    """Displays Confusion Matrix for model evaluation

    Args:
        true_labels (List): True values
        predicted_labels (List): Predicted Values
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(conf_matrix)


def print_evaluation_report(true_labels: List, predicted_labels: List):
    """Displays Classification Report for model evaluation

    Args:
        true_labels (List): True values
        predicted_labels (List): Predicted Values
    """
    report = classification_report(true_labels, predicted_labels)

    # Micro average metrics
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="micro"
    )

    # Append Micro Avg Metrics
    report += f"   micro avg       {micro_precision:.2f}      {micro_recall:.2f}      {micro_f1_score:.2f}       {len(true_labels)}"

    print(report)


def full_evaluation(analysis_type: str):
    """Perform evaluation of AI tools based on the specified type of analysis.

    Args:
        analysis_type (str): Type of analysis, e.g., "claim-detection" or "stance-detection".
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if analysis_type == "claim-detection":
        true_data_source_dir = os.path.join(current_dir, "output", "crowd-work", "claim-detection-ground-truth")
        ai_data_source_dir = os.path.join(current_dir, "output", "ai-prediction", "claim-detection")
        true_label_col = "is_check_worthy_claim"
        factiverse_label_col = "factiverse_is_CW"
        gpt4_label_col = "openai_gpt4_is_CW"
    elif analysis_type == "stance-detection":
        true_data_source_dir = os.path.join(current_dir, "output", "crowd-work", "stance-detection-ground-truth")
        ai_data_source_dir = os.path.join(current_dir, "output", "ai-prediction", "stance-detection")
        true_label_col = "stance"
        factiverse_label_col = "factiverse_stance"
        gpt4_label_col = "openai_gpt4_stance"
    else:
        print("UNKNOWN Analysis Type")
        return

    true_labels = []
    factiverse_labels = []
    openai_gpt4_labels = []

    print("------------------------------------------------------------------------------------------")
    # Fetch True Labels from Ground Truth Files
    for podcast_folder in os.listdir(true_data_source_dir):
        podcast_folder_path = os.path.join(true_data_source_dir, podcast_folder)

        if os.path.isdir(podcast_folder_path):

            for episode_file in os.listdir(podcast_folder_path):
                epiode_file_path = os.path.join(podcast_folder_path, episode_file)

                df = pd.read_csv(epiode_file_path)
                true_labels.extend(df[true_label_col].tolist())
                print(f"Fetched {len(df[true_label_col])} true label values from file {episode_file}.")

    print("------------------------------------------------------------------------------------------")

    # Fetch Predicted Labels from AI Prediction Files
    for podcast_folder in os.listdir(ai_data_source_dir):
        podcast_folder_path = os.path.join(ai_data_source_dir, podcast_folder)

        if os.path.isdir(podcast_folder_path):

            for episode_file in os.listdir(podcast_folder_path):
                epiode_file_path = os.path.join(podcast_folder_path, episode_file)

                df = pd.read_csv(epiode_file_path)
                factiverse_labels.extend(df[factiverse_label_col].tolist())
                openai_gpt4_labels.extend(df[gpt4_label_col].tolist())
                print(f"Fetched {len(df[factiverse_label_col])} predicted label values from file {episode_file}.")

    # Print total values to confirm equal number of labels
    print("------------------------------------------------------------------------------------------")
    print(f"Number of True Labels: {len(true_labels)}")
    print(f"Number of Factiverse Predicted Labels: {len(factiverse_labels)}")
    print(f"Number of OpenAI GPT4 Predicted Labels: {len(openai_gpt4_labels)}")

    # Evaluate performance of AI tools using extracted labels
    print("------------------------------------------------------------------------------------------")
    print(f"Performance Evaluation of Factiverse AI Tool for {analysis_type} in Podcasts")
    print_evaluation_report(true_labels, factiverse_labels)

    print("------------------------------------------------------------------------------------------")
    print(f"Performance Evaluation of OpenAI GPT4 Tool for {analysis_type} in Podcasts")
    print_evaluation_report(true_labels, openai_gpt4_labels)

    print("------------------------------------------------------------------------------------------")


def test_dataset_evaluation(analysis_type: str):
    """Perform evaluation of AI tools based on the specified type of analysis.

    Args:
        analysis_type (str): Type of analysis, e.g., "claim-detection" or "stance-detection".
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if analysis_type == "claim-detection":
        true_data_source_dir = os.path.join(current_dir, "output", "crowd-work", "claim-detection-ground-truth")
        true_test_pod_dir = utils.get_podcast_dir_path(true_data_source_dir, 672)
        true_test_file = os.path.join(true_test_pod_dir, "cd_ground_truth_ep_672.csv")

        ai_data_source_dir = os.path.join(current_dir, "output", "ai-prediction", "claim-detection")
        ai_test_pod_dir = utils.get_podcast_dir_path(ai_data_source_dir, 672)
        ai_test_file = os.path.join(ai_test_pod_dir, "ai_cd_ep_672.csv")

        true_label_col = "is_check_worthy_claim"
        factiverse_label_col = "factiverse_is_CW"
        gpt4_label_col = "openai_gpt4_is_CW"

    elif analysis_type == "stance-detection":
        true_data_source_dir = os.path.join(current_dir, "output", "crowd-work", "stance-detection-ground-truth")
        true_test_pod_dir = utils.get_podcast_dir_path(true_data_source_dir, 219)
        true_test_file = os.path.join(true_test_pod_dir, "sd_ground_truth_ep_219.csv")

        ai_data_source_dir = os.path.join(current_dir, "output", "ai-prediction", "stance-detection")
        ai_test_pod_dir = utils.get_podcast_dir_path(ai_data_source_dir, 219)
        ai_test_file = os.path.join(ai_test_pod_dir, "ai_sd_ep_219.csv")

        true_label_col = "stance"
        factiverse_label_col = "factiverse_stance"
        gpt4_label_col = "openai_gpt4_stance"
    else:
        print("UNKNOWN Analysis Type")
        return

    true_labels = []
    factiverse_labels = []
    openai_gpt4_labels = []

    print("------------------------------------------------------------------------------------------")

    df = pd.read_csv(true_test_file)
    true_labels = df[true_label_col].tolist()
    print(f"Fetched {len(df[true_label_col])} true label values from file {true_test_file}.")

    print("------------------------------------------------------------------------------------------")

    df = pd.read_csv(ai_test_file)
    factiverse_labels = df[factiverse_label_col].tolist()
    openai_gpt4_labels = df[gpt4_label_col].tolist()
    print(f"Fetched {len(df[factiverse_label_col])} predicted label values from file {ai_test_file}.")

    # Print total values to confirm equal number of labels
    print("------------------------------------------------------------------------------------------")
    print(f"Number of True Labels: {len(true_labels)}")
    print(f"Number of Factiverse Predicted Labels: {len(factiverse_labels)}")
    print(f"Number of OpenAI GPT4 Predicted Labels: {len(openai_gpt4_labels)}")

    # Evaluate performance of AI tools using extracted labels
    print("------------------------------------------------------------------------------------------")
    print(f"Performance Evaluation of Factiverse AI Tool for {analysis_type} in Podcasts")
    print_evaluation_report(true_labels, factiverse_labels)

    print("------------------------------------------------------------------------------------------")
    print(f"Performance Evaluation of OpenAI GPT4 Tool for {analysis_type} in Podcasts")
    print_evaluation_report(true_labels, openai_gpt4_labels)

    print("------------------------------------------------------------------------------------------")
