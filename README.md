# Podcast Fact-Checker

Podcast Fact-Checker is a tool designed to process podcast transcripts for detecting claims, assessing their stance, and generating fact-check summaries. This repository contains scripts and utilities to facilitate these tasks, including training models and evaluating their performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Main Operations](#main-operations)
  - [Training Models](#training-models)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/Podcast-factchecker.git
    cd Podcast-factchecker
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Main Operations

The main script provides a menu-driven interface to perform various operations such as generating ground truth datasets, evaluating model performance, and generating fact-check summaries. To run the main script:

```bash
python -m src.main
```

#### The menu will prompt you to choose an operation

1. Generate Ground Truth Dataset from Annotated Data Files
2. Evaluate Performance of Factiverse and OpenAI GPT-4
3. Generate Fact-Check Summary

#### Generate Ground Truth Dataset from Annotated Data Files

1. Choose the dataset type (Claim Detection or Stance Detection).
2. Select whether to process all episodes or a single episode.
3. The script will generate the ground truth dataset based on your selection.

#### Evaluate Performance of Factiverse and OpenAI GPT-4

1. Choose the dataset type (Claim Detection or Stance Detection).
2. Select the evaluation type (Full Evaluation or Test Dataset Evaluation).
3. The script will evaluate the performance based on your selection.

#### Generate Fact-Check Summary

1. Select whether to process all episodes or a single episode.
2. Choose the summary type (Short, Detailed, or Both).
3. The script will generate the fact-check summary based on your selection.

### Training Models

To train the models, find the respective directories and run the training scripts.

Examples:

1. Claim Detection Model (Albert)

    ```bash
    python -m src.llm_model_training.claim-detection.albert_cd
    ```

2. Stance Detection Model (DistilBERT)

    ```bash
    python -m src.llm_model_training.stance-detection.distilbert_sd
    ```

### Testing Models

To Test the fine-tuned transformer models run the following commands.

1. Claim Detection Model (Albert)

    ```bash
    python -m src.llm_model_training.claim-detection.fine_tuned_model_testing_cd
    ```

2. Stance Detection Model (DistilBERT)

    ```bash
    python -m src.llm_model_training.stance-detection.fine_tuned_model_testing_sd
    ```
