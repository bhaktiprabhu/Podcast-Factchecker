"""This module includes functions related to Claim Detection Phase"""

import os

import numpy as np
import pandas as pd

import openai_utils
from factiverse_api import factiverse_api
from utils import create_csv_from_df, execute_query_pandas, execute_query_sqlite, get_podcast_dir_path

current_dir = os.path.dirname(os.path.abspath(__file__))


def get_segmentation_id(episode_id: int) -> int:
    """Get Segmentation Id for an Episode

    Args:
        episode_id (int): Episode Id

    Returns:
        int: segmentation Id
    """
    q = f"""
            select id
            from api_segmentation
            where transcription_id = (
                select id
                from api_transcription
                where item_id = {episode_id}
            )
            and name = 'Factiverse CW/MO/CS'
        """
    seg_id = execute_query_sqlite(q, "one")[0]
    return seg_id


def get_utterances(episode_id: int) -> pd.DataFrame:
    """Fetch all utterances for an episode

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.DataFrame: Utterances for an Episode
    """
    seg_id = get_segmentation_id(episode_id)

    q = f"""
            select id, text, text_coref
            from api_utterance
            where segmentation_id = {seg_id}
        """

    utterances = execute_query_sqlite(q)
    utterance_df = pd.DataFrame(utterances, columns=["utterance_id", "text", "text_coref"])
    utterance_df["text_coref"] = utterance_df["text_coref"].fillna(utterance_df["text"])
    return utterance_df


def get_factiverse_claim_df(episode_id: int) -> pd.DataFrame:
    """Gets Factiverse Claim Detection Results for an episode transcription

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.DataFrame: Factiverse Claim Detection Output for an Episode
    """
    utterance_df = get_utterances(episode_id)

    facti_claim_df = utterance_df[["utterance_id", "text_coref"]]
    facti_claim_df.is_copy = False
    facti_claim_df = facti_claim_df.rename(columns={"text_coref": "utterance_text"})

    for i, text in enumerate(utterance_df["text_coref"]):
        facti_claim_check = factiverse_api.claim_detection(text)
        if len(facti_claim_check["detectedClaims"]) == 0:
            facti_claim_df.loc[i, "factiverse_is_CW"] = "FALSE"
        else:
            facti_claim_df.loc[i, "factiverse_is_CW"] = "TRUE"

    return facti_claim_df


def get_openai_claim_df(episode_id: int) -> pd.DataFrame:
    """Gets OpenAI Claim Detection Results for an episode transcription

    Args:
        episode_id (int): Episode Id

    Returns:
        pd.DataFrame: OpenAI Claim Detection Output for an Episode
    """
    utterance_df = get_utterances(episode_id)

    openai_claim_df = utterance_df[["utterance_id", "text_coref"]]
    openai_claim_df.is_copy = False
    openai_claim_df = openai_claim_df.rename(columns={"text_coref": "utterance_text"})

    for i, text in enumerate(utterance_df["text_coref"]):
        openai_claim_check = openai_utils.claim_detection(text)
        if openai_claim_check in ("No", "no"):
            openai_claim_df.loc[i, "openai_gpt4_is_CW"] = "FALSE"
        elif openai_claim_check in ("Yes", "yes"):
            openai_claim_df.loc[i, "openai_gpt4_is_CW"] = "TRUE"

    return openai_claim_df


def get_annotation_claim_df(episode_id: int) -> pd.DataFrame:
    """Gets annotation results for an episode transcription

    Args:
        episode_id (_type_): Episode Id

    Returns:
        pd.DataFrame: Annotation claims for an episode
    """
    seg_id = get_segmentation_id(episode_id)
    q = f"""
            select
                utterance_id,
                sum(case when category = 'Checkable' then 1 else 0 end) as CW_claim_count,
                sum(case when category = 'Not Checkable' then 1 else 0 end) as Not_CW_claim_count
            from toloka_annotation_df
            where qualifier = 'Checkworthiness'
            and segmentation_id = {seg_id}
            group by utterance_id
            """
    annot_claim_df = execute_query_pandas(q)
    annot_claim_df["toloka_is_cw"] = np.where(
        annot_claim_df["CW_claim_count"] < annot_claim_df["Not_CW_claim_count"], False, True
    )

    annot_claim_df["utterance_id"] = annot_claim_df["utterance_id"].astype(int)

    return annot_claim_df


def generate_toloka_annot_claim_detection_csv(episode_id: int):
    """Generates csv for an episode with analysis

    Args:
        episode_id (int): Episode Id
    """
    utterance_df = get_utterances(episode_id)
    utterance_df = utterance_df[["utterance_id", "text_coref"]]
    utterance_df.is_copy = False
    utterance_df = utterance_df.rename(columns={"text_coref": "utterance_text"})

    annot_claim_df = get_annotation_claim_df(episode_id)

    result_df = pd.merge(utterance_df, annot_claim_df, on="utterance_id", how="left")

    result_df["CW_claim_count"] = result_df["CW_claim_count"].infer_objects(copy=False).fillna(0)
    result_df["Not_CW_claim_count"] = result_df["Not_CW_claim_count"].infer_objects(copy=False).fillna(0)

    result_df["CW_claim_count"] = result_df["CW_claim_count"].astype(int)
    result_df["Not_CW_claim_count"] = result_df["Not_CW_claim_count"].astype(int)

    result_df = result_df.sort_values(by="utterance_id")
    result_df = result_df.reset_index(drop=True)

    output_path = os.path.join(current_dir, "output", "toloka-annotation")
    output_file_name = f"toloka_annot_cd_ep_{episode_id}.csv"

    create_csv_from_df(result_df, episode_id, output_path, output_file_name)


def generate_claim_detection_crowdwork_csv(episode_id: int):
    """Generates csv for getting true value for claim detection by crowd-work for an episode

    Args:
        episode_id (int): Episode Id
    """
    utterance_df = get_utterances(episode_id)

    crowd_work_df = utterance_df[["utterance_id", "text_coref"]]
    crowd_work_df.is_copy = False
    crowd_work_df = crowd_work_df.rename(columns={"text_coref": "utterance_text"})
    crowd_work_df["is_check_worthy_claim"] = "FALSE"
    crowd_work_df["relevance_level"] = None

    crowd_work_df = crowd_work_df.sort_values(by="utterance_id")
    crowd_work_df = crowd_work_df.reset_index(drop=True)

    output_path = os.path.join(current_dir, "output", "crowd-work", "claim-detection")
    output_file_name = f"crowd_work_cd_ep_{episode_id}.csv"

    create_csv_from_df(crowd_work_df, episode_id, output_path, output_file_name)


def generate_ai_claim_detection_prediction_csv(episode_id: int):
    """Generates csv with AI prediction for claim detection for an episode

    Args:
        episode_id (int): Episode Id
    """
    facti_claim_df = get_factiverse_claim_df(episode_id)
    openai_claim_df = get_openai_claim_df(episode_id)
    openai_claim_df = openai_claim_df.drop(columns=["utterance_text"])
    result_df = pd.merge(facti_claim_df, openai_claim_df, on="utterance_id", how="left")

    result_df = result_df.sort_values(by="utterance_id")
    result_df = result_df.reset_index(drop=True)

    output_path = os.path.join(current_dir, "output", "ai-prediction", "claim-detection")
    output_file_name = f"ai_cd_ep_{episode_id}.csv"

    create_csv_from_df(result_df, episode_id, output_path, output_file_name)


def generate_claim_detection_ground_truth_csv(episode_id: int):
    """Generates csv with Ground Truth for claim detection for an episode from crowd annotation work

    Args:
        episode_id (int): Episode Id
    """

    utterance_df = get_utterances(episode_id)

    cd_ground_truth_df = utterance_df[["utterance_id", "text_coref"]]
    cd_ground_truth_df.is_copy = False
    cd_ground_truth_df = cd_ground_truth_df.rename(columns={"text_coref": "utterance_text"})

    # Get Toloka Annotated Data
    toloka_is_cw = get_annotation_claim_df(episode_id)["toloka_is_cw"]
    cd_ground_truth_df["Toloka_annotation"] = toloka_is_cw

    relevance_df = cd_ground_truth_df[["utterance_id"]]

    data_source_path = os.path.join(current_dir, "data", "crowd-work", "claim-detection")
    podcast_dir = get_podcast_dir_path(data_source_path, episode_id)

    # Get Crowd-work annotated data from CSV
    for data_file_name in os.listdir(podcast_dir):
        if data_file_name.endswith(f"_cd_{episode_id}.csv"):
            data_file_path = os.path.join(podcast_dir, data_file_name)
            df = pd.read_csv(data_file_path)
            df = df.drop(columns=["utterance_text"])

            rel_col_name = data_file_name.split("_")[0] + "_relevance"
            relevance_df = relevance_df.copy()
            relevance_df.loc[:, rel_col_name] = df["relevance_level"]
            df = df.drop(columns=["relevance_level"])

            cw_col_name = data_file_name.split("_")[0] + "_annotation"

            df = df.rename(columns={"is_check_worthy_claim": cw_col_name})
            cd_ground_truth_df = pd.merge(cd_ground_truth_df, df, on="utterance_id", how="left")

    is_cw_modes = cd_ground_truth_df.iloc[:, 2:].mode(axis=1).iloc[:, 0]
    cd_ground_truth_df["is_check_worthy_claim"] = is_cw_modes

    rel_modes = relevance_df.iloc[:, 2:].mode(axis=1).iloc[:, 0]
    cd_ground_truth_df["relevance_level"] = rel_modes

    cd_ground_truth_df = cd_ground_truth_df[
        ["utterance_id", "utterance_text", "is_check_worthy_claim", "relevance_level"]
    ]

    output_path = os.path.join(current_dir, "output", "crowd-work", "claim-detection-ground-truth")
    output_file_name = f"cd_ground_truth_ep_{episode_id}.csv"
    create_csv_from_df(cd_ground_truth_df, episode_id, output_path, output_file_name)
