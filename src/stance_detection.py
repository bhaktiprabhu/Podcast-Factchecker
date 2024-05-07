"""This module includes functions related to Stance Detection Phase"""

import os

import pandas as pd

import openai_utils
from factiverse_api import factiverse_api
from utils import create_csv_from_df, get_podcast_dir_path

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
    podcast_dir = get_podcast_dir_path(data_source_path, episode_id)
    filename = f"cd_ground_truth_ep_{episode_id}.csv"
    file_path = os.path.join(podcast_dir, filename)

    cd_df = pd.read_csv(file_path)

    cw_claim_list = cd_df.loc[cd_df["is_check_worthy_claim"], "utterance_text"]

    print(type(cw_claim_list))

    return cw_claim_list


def generate_stance_detection_csv(episode_id: int):
    """Generates 2 CSV files for an episode:
        1. For getting true value for stance detection by crowd-work
        2. With AI prediction for stance detection

    Args:
        episode_id (int): Episode Id
    """

    crowd_work_df = pd.DataFrame(
        columns=["check_worthy_claim", "evidence_snippet", "article_url", "unresolved_coref", "stance"]
    )

    # Store Factiverse stance prediction in separate csv
    ai_stance_df = pd.DataFrame(
        columns=["check_worthy_claim", "evidence_snippet", "article_url", "factiverse_stance", "openai_gpt4_stance"]
    )

    for claim in get_cw_claims_list(episode_id):
        facti_stance = factiverse_api.stance_detection(claim)
        evidences = facti_stance["evidence"]

        for evidence in evidences:
            crowd_work_df.loc[len(crowd_work_df)] = {
                "checkworthy_claim": claim,
                "evidence_snippet": evidence["evidenceSnippet"],
                "article_url": evidence["url"],
                "unresolved_coref": False,
                "stance": "",
            }

            openai_stance_res = openai_utils.stance_detection(claim, evidence["evidenceSnippet"])

            ai_stance_df.loc[len(ai_stance_df)] = {
                "checkworthy_claim": claim,
                "evidence_snippet": evidence["evidenceSnippet"],
                "article_url": evidence["url"],
                "factiverse_stance": evidence["labelDescription"],
                "openai_gpt4_stance": openai_stance_res,
            }

    # Save the DataFrames to a CSV files

    output_path = os.path.join(current_dir, "output", "crowd-work", "stance-detection")
    output_file_name = f"crowd_work_sd_ep_{episode_id}.csv"

    create_csv_from_df(crowd_work_df, episode_id, output_path, output_file_name)

    output_path = os.path.join(current_dir, "output", "ai-prediction", "stance-detection")
    output_file_name = f"ai_sd_ep_{episode_id}.csv"

    create_csv_from_df(ai_stance_df, episode_id, output_path, output_file_name)

