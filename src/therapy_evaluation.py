import numpy as np
import pandas as pd
import pickle
from src.therapy_mdp import Therapy_MDP
# from therapy_mdp import Therapy_MDP
from joblib import Parallel, delayed
from copy import deepcopy
import click
from scipy.stats import entropy
import json

def helper(tmdp, patient_id, num_of_cf_trajectories, policy):

    summaries = []
    A_real = tmdp.get_trajectory_actions(patient_id=patient_id)

    P_cf = tmdp.get_counterfactual_MDP(patient_id=patient_id, recompute=False)
    horizon = np.max([x[1] for x in list(P_cf.keys())]) + 1

    # Compute average counterfactual outcome
    for k in range(horizon+1):
        summary = {}
        summary['id'] = int(patient_id)

        summary['k'] = k
        summary['horizon'] = int(horizon)
        summary['policy'] = policy
        summary['seeds'] = {}
        if policy == 'optimal':
            pi, _ = tmdp.maximize(patient_id=patient_id, k=k)
            for seed in range(1, num_of_cf_trajectories+1):
                _, _, _, reward, _ = tmdp.sample_cf_trajectory(patient_id=patient_id, pi=pi, seed=seed)
                summary['seeds'][seed] = str(reward)
        else:
            for seed in range(1, num_of_cf_trajectories+1):
                _, _, a, reward, _ = tmdp.sample_cf_trajectory(patient_id=patient_id, pi=policy, k=k, seed=seed)
                dist = sum(1 for ind, act in enumerate(a) if a[ind] != A_real[ind])
                assert dist<=k, 'Problem with counterfactual trajectory'
                summary['seeds'][seed] = str(reward)
        
        summaries.append(summary)

    print('Done ID '+str(patient_id))
    return summaries

@click.command() # Comment the click commands for testing
@click.option('--num_of_cf_samples', type=int, required=True, help="Number of counterfactual samples")
@click.option('--n_jobs', type=int, required=True, help="Number of parallel threads")
@click.option('--min_horizon', type=int, required=True, help="Minimum horizon")
@click.option('--outputs', type=str, required=True, help="Output directory for counterfactual MDPs")
@click.option('--data', type=str, required=True, help="File containing the therapy data")
@click.option('--cf_mdp_directory', type=str, required=True, help="Output directory for counterfactual MDPs")
@click.option('--num_of_cf_trajectories', type=int, required=True, help="Number of counterfactual explanations")
@click.option('--policy', type=str, required=True, help="Policy to be used")
def therapy_evaluation(num_of_cf_samples, cf_mdp_directory, n_jobs, outputs, num_of_cf_trajectories, min_horizon, data, policy):

    tmdp = Therapy_MDP(data_filename=data, cf_mdp_directory=cf_mdp_directory, min_horizon=min_horizon)
    tmdp.initialize_MDP(unobserved_reward='normal')
    patient_IDs = tmdp.get_valid_IDs()

    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(helper)(tmdp=tmdp, patient_id=patient_id, num_of_cf_trajectories=num_of_cf_trajectories, policy=policy) for patient_id in patient_IDs)
    final_results = [item for sublist in results for item in sublist]
    for summary in final_results:
        with open(outputs+'therapy_evaluation_id_' + str(summary['id']) + '_k_' + str(summary['k']) + '_policy_' + str(policy) +'.json', 'w') as f:
            json.dump(summary, f)

if __name__ == '__main__':
    therapy_evaluation()
    # therapy_evaluation(num_of_cf_samples=1000, cf_mdp_directory='outputs/cf_mdps/', n_jobs=1, outputs='outputs/',
    #             num_of_cf_trajectories=100, min_horizon=10, data='data/therapy/therapy.csv', policy='greedy')