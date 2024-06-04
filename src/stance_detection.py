"""This module includes functions related to Stance Detection Phase"""

import os
import random

import pandas as pd
from requests import ReadTimeout

from src import factiverse_utils, openai_utils, utils

current_dir = os.path.dirname(os.path.abspath(__file__))


def get_cw_claims_list(episode_id: int) -> pd.Series:
    """Get list of utterances marked as check-worthy claims from crowd-work annotation

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.Series: A list of utterances marked as check-worthy claims
    """
    # Getting the episode file
    data_source_path = os.path.join(current_dir, "output", "crowd-work", "claim-detection-ground-truth")
    podcast_dir = utils.get_podcast_dir_path(data_source_path, episode_id)
    filename = f"cd_ground_truth_ep_{episode_id}.csv"
    file_path = os.path.join(podcast_dir, filename)

    cd_df = pd.read_csv(file_path)

    cw_claim_list = cd_df.loc[cd_df["is_check_worthy_claim"], "utterance_text"]

    return cw_claim_list


def generate_stance_detection_csv(episode_id: int):
    """Generates 2 CSV files for an episode:
        1. For getting true value for stance detection by crowd-work
        2. With AI prediction for stance detection

    Args:
        episode_id (int): Episode Id
    """

    crowd_work_df = pd.DataFrame(
        columns=["check_worthy_claim", "evidence_snippet", "article_url", "is_unresolved_coref", "stance"]
    )

    # Store Factiverse stance prediction in separate csv
    ai_stance_df = pd.DataFrame(
        columns=["check_worthy_claim", "evidence_snippet", "article_url", "factiverse_stance", "openai_gpt4_stance"]
    )

    cw_claims_list = get_cw_claims_list(episode_id)
    print("Total number of Chech-worthy claims for Evidence Retrieval: ", len(cw_claims_list))

    for claim in cw_claims_list:

        try:
            facti_stance = factiverse_utils.factiverse_api.stance_detection(claim)
        except ReadTimeout:
            # Exclude claims that timed out while searching for evidence
            # Most cases are the one's with unresolved coreference resolution
            print("ReadTimeout Error Occured for Claim: ", claim)
            continue

        evidences = facti_stance["evidence"]

        evidences = random.sample(evidences, 5) if len(evidences) > 5 else evidences

        for evidence in evidences:
            crowd_work_df.loc[len(crowd_work_df)] = {
                "check_worthy_claim": claim,
                "evidence_snippet": evidence["evidenceSnippet"],
                "article_url": evidence["url"],
                "is_unresolved_coref": False,
                "stance": "",
            }

            openai_stance_res = openai_utils.stance_detection(claim, evidence["evidenceSnippet"])

            ai_stance_df.loc[len(ai_stance_df)] = {
                "check_worthy_claim": claim,
                "evidence_snippet": evidence["evidenceSnippet"],
                "article_url": evidence["url"],
                "factiverse_stance": evidence["labelDescription"],
                "openai_gpt4_stance": openai_stance_res,
            }

    # Save the DataFrames to a CSV files

    output_path = os.path.join(current_dir, "output", "crowd-work", "stance-detection")
    output_file_name = f"crowd_work_sd_ep_{episode_id}.csv"

    utils.create_csv_from_df(crowd_work_df, episode_id, output_path, output_file_name)

    output_path = os.path.join(current_dir, "output", "ai-prediction", "stance-detection")
    output_file_name = f"ai_sd_ep_{episode_id}.csv"

    utils.create_csv_from_df(ai_stance_df, episode_id, output_path, output_file_name)


def generate_stance_detection_ground_truth_csv(episode_id: int):
    """Generates csv with Ground Truth for stance detection for an episode from crowd annotation work

    Args:
        episode_id (int): Episode Id
    """
    sd_ground_truth_df = None

    data_source_path = os.path.join(current_dir, "data", "crowd-work", "stance-detection")
    podcast_dir = utils.get_podcast_dir_path(data_source_path, episode_id)

    # Get Crowd-work annotated data from CSV
    for data_file_name in os.listdir(podcast_dir):
        if data_file_name.endswith(f"_sd_{episode_id}.csv"):

            data_file_path = os.path.join(podcast_dir, data_file_name)

            df = pd.read_csv(data_file_path)
            coref_col_name = data_file_name.split("_")[0] + "_coref"
            stance_col_name = data_file_name.split("_")[0] + "_stance"

            df = df.rename(columns={"is_unresolved_coref": coref_col_name, "stance": stance_col_name})

            # For first matched csv
            if sd_ground_truth_df is None:
                sd_ground_truth_df = df.drop(columns=coref_col_name)
                coref_df = df[[coref_col_name]]
            else:
                coref_df = pd.concat([coref_df, df[[coref_col_name]]], axis=1)
                df = df.drop(columns=["check_worthy_claim", "evidence_snippet", "article_url", coref_col_name])
                sd_ground_truth_df = pd.concat([sd_ground_truth_df, df], axis=1)

    is_unresolved_coref_modes = coref_df.mode(axis=1).iloc[:, 0]
    stance_modes = sd_ground_truth_df.iloc[:, 3:].mode(axis=1).iloc[:, 0]

    sd_ground_truth_df["is_unresolved_coref"] = is_unresolved_coref_modes
    sd_ground_truth_df["stance"] = stance_modes

    sd_ground_truth_df = sd_ground_truth_df[
        ["check_worthy_claim", "evidence_snippet", "article_url", "stance", "is_unresolved_coref"]
    ]

    # Filter out rows where "is_unresolved_coref" is True and drop the column
    sd_ground_truth_df = sd_ground_truth_df[sd_ground_truth_df["is_unresolved_coref"] == False]
    sd_ground_truth_df = sd_ground_truth_df.drop(columns=["is_unresolved_coref"])

    # Filter out rows where "is_unresolved_coref" is True and drop the column
    sd_ground_truth_df = sd_ground_truth_df[sd_ground_truth_df["stance"] != "#CHOOSE#"]

    output_path = os.path.join(current_dir, "output", "crowd-work", "stance-detection-ground-truth")
    output_file_name = f"sd_ground_truth_ep_{episode_id}.csv"
    utils.create_csv_from_df(sd_ground_truth_df, episode_id, output_path, output_file_name)


def filter_ai_stance_dectection(episode_id: int):
    """Filter the generated AI Prediction dataframe to elimainate the rows which do not exist in ground truth dataset.
       This is to remove the rows where the claims do not have coreference resolved.

    Args:
        episode_id (int): Episode Id
    """
    true_data_source_dir = os.path.join(current_dir, "output", "crowd-work", "stance-detection-ground-truth")
    true_podcast_dir = utils.get_podcast_dir_path(true_data_source_dir, episode_id)
    ai_data_source_dir = os.path.join(current_dir, "output", "ai-prediction", "stance-detection")
    ai_podcast_dir = utils.get_podcast_dir_path(ai_data_source_dir, episode_id)

    # Load the dataframes
    true_df = pd.read_csv(os.path.join(true_podcast_dir, f"sd_ground_truth_ep_{episode_id}.csv"))
    ai_df = pd.read_csv(os.path.join(ai_podcast_dir, f"ai_sd_ep_{episode_id}.csv"))

    # Create a set of tuples for (check_worthy_claim, evidence_snippet) from the ground truth dataframe
    true_claims_evidences = set(zip(true_df["check_worthy_claim"], true_df["evidence_snippet"]))

    # Filter the AI predictions dataframe
    ai_df_filtered = ai_df[
        ai_df.apply(lambda row: (row["check_worthy_claim"], row["evidence_snippet"]) in true_claims_evidences, axis=1)
    ]

    # Print the shapes of the original and filtered dataframes for verification
    print("True DF Shape:", true_df.shape)
    print("Original AI predictions shape:", ai_df.shape)
    print("Filtered AI predictions shape:", ai_df_filtered.shape)

    output_path = os.path.join(current_dir, "output", "ai-prediction", "stance-detection")
    output_file_name = f"ai_sd_ep_{episode_id}.csv"

    utils.create_csv_from_df(ai_df_filtered, episode_id, output_path, output_file_name)


generate_stance_detection_ground_truth_csv(219)
# filter_ai_stance_dectection(630)
