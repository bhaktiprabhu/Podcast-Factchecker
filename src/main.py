"""This file facilitates a user menu to perform certain operations"""

from src.ai_tools_performance import full_evaluation, test_dataset_evaluation
from src.claim_detection import generate_claim_detection_ground_truth_csv
from src.fact_check_summary import generate_fact_check_summary
from src.stance_detection import generate_stance_detection_ground_truth_csv
from src.utils import list_all_episodes


def print_dataset_options():
    """prints dataset options"""
    print("-----------------------------------")
    print("Choose Dataset Type:-")
    print("1. Claim Detection")
    print("2. Stance Detection")


def print_episode_choice():
    """prints episode choices"""
    print("----------------------------------------------------------------------")
    print("Select whether to perform operation on single episode or all available episodes:-")
    print("1. All Episodes")
    print("2. Single Episode")


if __name__ == "__main__":
    print("----------------------------------------------------------------------")
    print("List of operation:-")
    print("1. Generate Ground Truth Dataset from Annotated Data Files")
    print("2. Evaluate Performance of Factiverse and OpenAI GPT4")
    print("3. Generate Fact-Check Summary")

    opt = int(input("Select an operation to perform: "))

    # Generate Ground Truth Dataset from Annotated Data Files
    if opt == 1:
        print_dataset_options()
        dataset_type = int(input("Enter choice number: "))

        print_episode_choice()
        ep_choice = int(input("Enter choice number: "))

        if dataset_type == 1:  # Claim Detection
            if ep_choice == 1:  # All episodes
                ep_ids = list_all_episodes()
                print(ep_ids)
                for ep_id in ep_ids:
                    generate_claim_detection_ground_truth_csv(ep_id)

            elif ep_choice == 2:
                ep_id = int(input("Enter the Episode Id: "))
                generate_claim_detection_ground_truth_csv(ep_id)

            else:
                raise ValueError("Invalid Choice Entered")

        elif dataset_type == 2:  # Stance Detection
            if ep_choice == 1:  # All episodes
                ep_ids = list_all_episodes()
                print(ep_ids)
                for ep_id in ep_ids:
                    generate_stance_detection_ground_truth_csv(ep_id)
            elif ep_choice == 2:
                ep_id = int(input("Enter the Episode Id: "))
                generate_stance_detection_ground_truth_csv(ep_id)
            else:
                raise ValueError("Invalid Choice Entered")

    # Evaluate Performance of Factiverse and OpenAI GPT4
    elif opt == 2:
        print_dataset_options()
        dataset_type = int(input("Enter choice number: "))

        print("----------------------------------------------------------------------")
        print("Choose Evaluation Type:-")
        print("1. Perform Evaluation for Entire Dataset")
        print("2. Perform Evaluation for Test Dataset")
        eval_type = int(input("Enter choice number: "))

        if eval_type == 1:  # Full evaluation
            if dataset_type == 1:  # Claim Detection
                full_evaluation("claim-detection")
            elif dataset_type == 2:  # Stance Detection
                full_evaluation("stance-detection")
            else:
                raise ValueError("Invalid Choice Entered")

        elif eval_type == 2:  # Test evaluation
            if dataset_type == 1:
                test_dataset_evaluation("claim-detection")
            elif dataset_type == 2:  # Stance Detection
                test_dataset_evaluation("stance-detection")
            else:
                raise ValueError("Invalid Choice Entered")
        else:
            raise ValueError("Invalid Choice Entered")

    # Generate Fact-Check Summary for an Episode
    elif opt == 3:
        print_episode_choice()
        ep_choice = int(input("Enter choice number: "))

        print("----------------------------------------------------------------------")
        print("Choose Summary Type:-")
        print("1. Short")
        print("2. Detailed")
        print("3. Both")
        summary_choice = int(input("Enter choice number: "))
        summary_type = {1: "short", 2: "detailed", 3: "both"}

        if ep_choice == 1:  # All episodes
            ep_ids = list_all_episodes()
            print(ep_ids)
            for ep_id in ep_ids:
                generate_fact_check_summary(ep_id, summary_type[summary_choice])

        elif ep_choice == 2:
            ep_id = int(input("Enter the Episode Id: "))
            generate_fact_check_summary(ep_id, summary_type[summary_choice])
        else:
            raise ValueError("Invalid Choice Entered")
