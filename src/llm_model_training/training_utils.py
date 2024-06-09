"""This file contains utility function for transformer models training and testing"""

import os

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction, Trainer

from src import utils


class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance in classification tasks."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Computes the focal loss between `inputs` and `targets`."""
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)  # pylint: disable=invalid-unary-operand-type
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalTrainer(Trainer):
    """Custom Trainer class that uses Focal Loss for training."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Class frequencies
        alpha = self.class_weights.to(logits.device)

        # Custom Focal Loss function
        loss_fct = FocalLoss(alpha=alpha, gamma=2.5)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    """Custom Trainer class to compute custom loss"""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Compute metrics function for evaluation
def compute_metrics(p: EvalPrediction):
    """Compute accuracy, precision, recall, and F1 score for evaluation predictions.

    Args:
        p (EvalPrediction): An object containing predictions and true labels.

    Returns:
        dict: A dictionary with the computed accuracy, precision, recall, and F1 score.
    """
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    # probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1)
    # preds = (probs[:, 1] > 0.3).int()  # Adjust threshold here
    # labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_dataset(dataset_type: str) -> DatasetDict:
    """Creates dataset with train. validation and test splits

    Args:
        dataset_type (str): Claim Detection or Stance Detection

    Raises:
        ValueError: For Invalid dataset type

    Returns:
        DatasetDict: Dictionary of Dataset splits
    """
    # Get Train and Validation Files
    eps = utils.list_all_episodes()
    # eps = [219, 555, 562, 563, 564, 565, 566, 630, 672]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.abspath(current_dir))

    train_files, validation_files, test_file = [], [], []

    if dataset_type == "claim-detection":
        data_source_path = os.path.join(parent_dir, "output", "crowd-work", "claim-detection-ground-truth")
        file_prefix = "cd_ground_truth_ep"

    elif dataset_type == "stance-detection":
        eps.reverse()
        data_source_path = os.path.join(parent_dir, "output", "crowd-work", "stance-detection-ground-truth")
        file_prefix = "sd_ground_truth_ep"

    else:
        raise ValueError("Invalid Dataset Type")

    for i, ep_id in enumerate(eps):
        podcast_dir = utils.get_podcast_dir_path(data_source_path, ep_id)
        filename = f"{file_prefix}_{ep_id}.csv"
        file_path = os.path.join(podcast_dir, filename)
        if i < 7:
            train_files.append(file_path)
        elif i < 9:
            validation_files.append(file_path)
        else:
            test_file.append(file_path)

    # print(train_files)
    # print(validation_files)
    # print(test_file)

    # Load the dataset
    dataset = load_dataset("csv", data_files={"train": train_files, "validation": validation_files, "test": test_file})

    if dataset_type == "claim-detection":
        dataset = dataset.remove_columns(["utterance_id"]).rename_column("is_check_worthy_claim", "label")
    else:
        dataset = dataset.remove_columns(["article_url"]).rename_column("stance", "label")

    print("---------------Dataset Loaded------------------")
    print(dataset)

    return dataset


def get_class_weights_train(num_class_0_samples: int, num_class_1_samples: int) -> torch.tensor:
    """Returns class weight tensor calculated using inverse frequency method

    Args:
        num_class_0_samples (int): Number of samples in class 0 (Not CW Claim/REFUTES)
        num_class_1_samples (int): Number of samples in class 1 (CW Claim/SUPPORTS)

    Returns:
        torch.tensor: class weights tensor
    """
    total_samples_train = num_class_0_samples + num_class_1_samples

    # Calculate the Class Weights Using Inverse Frequency
    weight_for_class_0_train = total_samples_train / num_class_0_samples
    weight_for_class_1_train = total_samples_train / num_class_1_samples

    print("Class Weights: ", [weight_for_class_0_train, weight_for_class_1_train])

    # Normalize Weights
    total_weight = weight_for_class_0_train + weight_for_class_1_train
    weight_class_0_normalized = weight_for_class_0_train / total_weight
    weight_class_1_normalized = weight_for_class_1_train / total_weight

    print("Class Weights Normalized: ", [weight_class_0_normalized, weight_class_1_normalized])

    class_weights_train = torch.tensor([weight_class_0_normalized, weight_class_1_normalized])

    return class_weights_train
