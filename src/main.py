"""This is the main file"""
from claim_analysis import generate_claim_comparison_csv
from model_performance import model_performance_evaluation
from utils import list_all_episodes

# ep_ids = list_all_episodes()
# print(ep_ids)

# for ep_id in ep_ids:
#     generate_claim_comparison_csv(ep_id)
#     print(f"Output generated for Episode {ep_id}")


model_performance_evaluation('abc')
