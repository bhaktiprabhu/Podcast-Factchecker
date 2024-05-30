"""Training of Distilbert Transformer model with Podcast Data"""

import os

import numpy as np
import torch
from datasets import load_dataset
from optimum.bettertransformer import BetterTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

from src import utils

current_dir = os.path.dirname(__file__)


# Preprocess the data
def tokenize_function(examples):
    """Tokenizes the input text data for model training or evaluation.

    Args:
        examples (dict): A dictionary containing the input text data to be tokenized.
                         It is expected to have a key "utterance_text" with the text to be tokenized.

    Returns:
        dict: A dictionary with tokenized input data, including input_ids, attention_mask, and potentially other tokenizer outputs.
    """
    return tokenizer(examples["utterance_text"], padding="max_length", truncation=True, max_length=256)


# Compute metrics function for evaluation
def compute_metrics(p: EvalPrediction):
    """Compute accuracy, precision, recall, and F1 score for evaluation predictions.

    Args:
        p (EvalPrediction): An object containing predictions and true labels.

    Returns:
        dict: A dictionary with the computed accuracy, precision, recall, and F1 score.
    """
    adjusted_preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    # preds = p.predictions.argmax(-1)
    # probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1)
    # adjusted_preds = (probs[:, 1] > 0.2).int()  # Adjust threshold here

    conf_matrix = confusion_matrix(labels, adjusted_preds)
    print(conf_matrix)

    report = classification_report(labels, adjusted_preds, target_names=["not_cw_claim", "is_cw_claim"])

    # Micro average metrics
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(
        labels, adjusted_preds, average="micro"
    )

    # Append Micro Avg Metrics
    report += f"   micro avg       {micro_precision:.2f}      {micro_recall:.2f}      {micro_f1_score:.2f}       {len(labels)}"

    print(report)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, adjusted_preds, average="weighted")
    acc = accuracy_score(labels, adjusted_preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


################################################################################################

# Get Train Test File (Episode Id:672)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.abspath(current_dir))

data_source_path = os.path.join(parent_dir, "output", "crowd-work", "claim-detection-ground-truth")

podcast_dir = utils.get_podcast_dir_path(data_source_path, 672)
filename = f"cd_ground_truth_ep_{672}.csv"
test_file_path = os.path.join(podcast_dir, filename)


# Load the test dataset
dataset = load_dataset("csv", data_files={"test": test_file_path})
print("---------------Test Dataset Loaded------------------")
print(dataset)

test_dataset = dataset["test"].remove_columns(["utterance_id"]).rename_column("is_check_worthy_claim", "label")

print("---------------Test Dataset------------------")
print(test_dataset)
print(test_dataset.shape)
print(test_dataset.size_in_bytes)


# Get the fine-tuned model directories
albert_model_directory = os.path.join(current_dir, "saved-models", "albert-focal")
distilbert_model_directory = os.path.join(current_dir, "saved-models", "distilbert-focal-2_5")
distilroberta_model_directory = os.path.join(current_dir, "saved-models", "distilroberta-focal")
mobilebert_model_directory = os.path.join(current_dir, "saved-models", "mobilebert-focal")


# Load the tokenizers and Tokenize the dataset as per the model
tokenizer = AutoTokenizer.from_pretrained(albert_model_directory, do_lower_case=True)
albert_tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenizer = AutoTokenizer.from_pretrained(distilbert_model_directory, do_lower_case=True)
distilbert_tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenizer = AutoTokenizer.from_pretrained(distilroberta_model_directory, do_lower_case=True)
distilroberta_tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenizer = AutoTokenizer.from_pretrained(mobilebert_model_directory, do_lower_case=True)
mobilebert_tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


# Load the fine-tuned models
albert_podcast_cd_model = AutoModelForSequenceClassification.from_pretrained(albert_model_directory, num_labels=2)
distilbert_podcast_cd_model = AutoModelForSequenceClassification.from_pretrained(
    distilbert_model_directory, num_labels=2
)
distilroberta_podcast_cd_model = AutoModelForSequenceClassification.from_pretrained(
    distilroberta_model_directory, num_labels=2
)
mobilebert_podcast_cd_model = AutoModelForSequenceClassification.from_pretrained(
    mobilebert_model_directory, num_labels=2
)

# Tranform the models for Faster Inference
# AlBERT model gives error during prediction if using BetterTransformer
bt_distilbert_podcast_cd_model = BetterTransformer.transform(distilbert_podcast_cd_model)
bt_distilroberta_podcast_cd_model = BetterTransformer.transform(distilroberta_podcast_cd_model)
# MobileBERT is not supported by BetterTransformer


train_ouptut_dir = os.path.join(current_dir, "training-results")

# Training Args to Create a Trainer Object
training_args = TrainingArguments(
    output_dir=train_ouptut_dir,
    per_device_eval_batch_size=16,
)

# Create Trainers for Prediction and Evaluation
albert_podcast_cd_trainer = Trainer(model=albert_podcast_cd_model, args=training_args, compute_metrics=compute_metrics)
distilbert_podcast_cd_trainer = Trainer(
    model=bt_distilbert_podcast_cd_model, args=training_args, compute_metrics=compute_metrics
)
distilroberta_podcast_cd_trainer = Trainer(
    model=bt_distilroberta_podcast_cd_model, args=training_args, compute_metrics=compute_metrics
)
mobilebert_podcast_cd_trainer = Trainer(
    model=mobilebert_podcast_cd_model, args=training_args, compute_metrics=compute_metrics
)

# Get Predictions and Evaluations for Testing the Model
print("-------------------------------------------------------------------------------------------")
print("**** AlBERT Podcast Claim Detection Model Evaluation for Test Data****")
albert_podcast_cd_predictions = albert_podcast_cd_trainer.predict(albert_tokenized_test_dataset)
print(albert_podcast_cd_predictions.metrics)

print("-------------------------------------------------------------------------------------------")
print("**** DistilBERT Podcast Claim Detection Model Evaluation for Test Data****")
distilbert_podcast_cd_predictions = distilbert_podcast_cd_trainer.predict(distilbert_tokenized_test_dataset)
print(distilbert_podcast_cd_predictions.metrics)

print("-------------------------------------------------------------------------------------------")
print("**** DistilRoBERTa Podcast Claim Detection Model Evaluation for Test Data****")
distilroberta_podcast_cd_predictions = distilroberta_podcast_cd_trainer.predict(distilroberta_tokenized_test_dataset)

print(distilroberta_podcast_cd_predictions.metrics)

print("-------------------------------------------------------------------------------------------")
print("**** MobileBERT Podcast Claim Detection Model Evaluation for Test Data****")
mobilebert_podcast_cd_predictions = mobilebert_podcast_cd_trainer.predict(mobilebert_tokenized_test_dataset)
print(mobilebert_podcast_cd_predictions.metrics)

print("-------------------------------------------------------------------------------------------")
