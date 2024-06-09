"""This module includes functions related to Generating Fact-check Summary for a Podcast Episode"""

import json
import os
from collections import Counter
from typing import Dict

import pandas as pd

from src import utils

current_dir = os.path.dirname(os.path.abspath(__file__))


def convert_to_markdown(summary: Dict) -> str:
    """Converts Summary to Markdown format

    Args:
        summary (Dict): Summary in Json Object Format

    Returns:
        str: Summary in Markdown Format
    """
    markdown_summary = f"# Fact-Check Summary for Episode {summary['episode_id']}\n\n"
    markdown_summary += "## Episode Information\n\n"
    markdown_summary += f"**Episode Name**: {summary['episode_name']}  \n"
    markdown_summary += f"**Podcast Name**: {summary['podcast_name']}  \n"
    markdown_summary += f"**Podcast Author**: {summary['podcast_author']}  \n"
    markdown_summary += f"**Podcast Categories**: {summary['podcast_categories']}\n\n"
    markdown_summary += "## Fact-Check Report\n\n"
    markdown_summary += f"- **Total Checkworthy Claims**: {summary['total_checkworthy_claims']}\n"
    markdown_summary += (
        f"- **Total Checkworthy Claims with Evidence**: {summary['total_checkworthy_claims_with_evidence']}\n"
    )
    markdown_summary += f"- **Claims Supported**: {summary['total_supported_claims']}\n"
    markdown_summary += f"- **Claims Refuted**: {summary['total_refuted_claims']}\n\n"

    markdown_summary += "### Supported Claims\n\n"
    for i, supported_claim in enumerate(summary["claims_supported"], 1):
        markdown_summary += f"{i}. {supported_claim}\n"

    markdown_summary += "\n### Refuted Claims\n\n"
    for i, refuted_claim in enumerate(summary["claims_refuted"], 1):
        markdown_summary += f"{i}. {refuted_claim}\n"

    return markdown_summary


def generate_fact_check_summary(episode_id, summary_type="both"):
    """Generates json and markdown files with fact-check summary for an episode

    Args:
        episode_id (int): Episode Id
        summary_type(str): detailed/short/both
    """
    ep_info = utils.get_episode_info(episode_id)

    # Get the CSV files
    data_source_path = os.path.join(current_dir, "output", "crowd-work", "claim-detection-ground-truth")
    podcast_dir = utils.get_podcast_dir_path(data_source_path, episode_id)
    claim_detection_csv = os.path.join(podcast_dir, f"cd_ground_truth_ep_{episode_id}.csv")

    data_source_path = os.path.join(current_dir, "output", "crowd-work", "stance-detection-ground-truth")
    podcast_dir = utils.get_podcast_dir_path(data_source_path, episode_id)
    stance_detection_csv = os.path.join(podcast_dir, f"sd_ground_truth_ep_{episode_id}.csv")

    # Read the CSV files
    claim_detection_df = pd.read_csv(claim_detection_csv)
    stance_detection_df = pd.read_csv(stance_detection_csv)

    # Filter checkworthy claims
    checkworthy_claims_df = claim_detection_df[claim_detection_df["is_check_worthy_claim"] == True]

    # Summary statistics
    total_checkworthy_claims = len(checkworthy_claims_df)
    stance_detection_grouped = stance_detection_df.groupby("check_worthy_claim")

    # Prepare detailed summary
    detailed_summary = {
        "episode_id": str(episode_id),
        "episode_name": ep_info[0],
        "podcast_name": ep_info[1],
        "podcast_author": ep_info[2],
        "podcast_categories": ep_info[3],
        "total_checkworthy_claims": total_checkworthy_claims,
        "list_checkworthy_claims": checkworthy_claims_df["utterance_text"].tolist(),
        "total_checkworthy_claims_with_evidence": stance_detection_grouped.ngroups,
        "list_checkworthy_claims_with_evidence": [],
        "overall_summary": {
            "total_supported_claims": 0,
            "total_refuted_claims": 0,
            "claims_supported": [],
            "claims_refuted": [],
        },
    }

    short_summary = {
        "episode_id": str(episode_id),
        "episode_name": ep_info[0],
        "podcast_name": ep_info[1],
        "podcast_author": ep_info[2],
        "podcast_categories": ep_info[3],
        "total_checkworthy_claims": total_checkworthy_claims,
        "total_checkworthy_claims_with_evidence": stance_detection_grouped.ngroups,
        "total_supported_claims": 0,
        "total_refuted_claims": 0,
        "claims_supported": [],
        "claims_refuted": [],
    }

    # Process each checkworthy claim with evidence
    for claim, group in stance_detection_grouped:
        claim_data = {"claim": claim, "retrieved_evidences": []}
        stance_counts = Counter()

        for _, row in group.iterrows():
            evidence_data = {
                "evidence_snippet": row["evidence_snippet"],
                "article_url": row["article_url"],
                "stance": row["stance"],
            }
            claim_data["retrieved_evidences"].append(evidence_data)
            stance_counts[row["stance"]] += 1

        # Determine overall stance
        if stance_counts["SUPPORTS"] > stance_counts["REFUTES"]:
            short_summary["claims_supported"].append(claim)
            detailed_summary["overall_summary"]["claims_supported"].append(claim)
        else:
            short_summary["claims_refuted"].append(claim)
            detailed_summary["overall_summary"]["claims_refuted"].append(claim)

        detailed_summary["list_checkworthy_claims_with_evidence"].append(claim_data)

    short_summary["total_supported_claims"] = len(short_summary["claims_supported"])
    short_summary["total_refuted_claims"] = len(short_summary["claims_refuted"])
    detailed_summary["overall_summary"]["total_supported_claims"] = len(
        detailed_summary["overall_summary"]["claims_supported"]
    )
    detailed_summary["overall_summary"]["total_refuted_claims"] = len(
        detailed_summary["overall_summary"]["claims_refuted"]
    )

    output_path = os.path.join(current_dir, "output", "fact-check-summary")
    podcast_dir = utils.get_podcast_dir_path(output_path, episode_id)

    # Save JSON summaries
    if summary_type in ["detailed", "both"]:
        output_file_name = f"fact_check_summary_ep_{episode_id}_detailed.json"
        output_file_path = os.path.join(podcast_dir, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json.dump(detailed_summary, json_file, indent=4)
        print(f" Detailed Summary generated for Episode {episode_id} at {output_file_path}")

    if summary_type in ["short", "both"]:
        output_file_name = f"fact_check_summary_ep_{episode_id}_short.json"
        output_file_path = os.path.join(podcast_dir, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as json_file:
            json.dump(short_summary, json_file, indent=4)
        print(f" Short Summary generated for Episode {episode_id} at {output_file_path}")

        # Convert to Markdown and save
        markdown_summary = convert_to_markdown(short_summary)
        output_file_name = f"fact_check_summary_ep_{episode_id}_short.md"
        output_file_path = os.path.join(podcast_dir, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_summary)

        print(f" Short Summary in Markdown Format generated for Episode {episode_id} at {output_file_path}")
