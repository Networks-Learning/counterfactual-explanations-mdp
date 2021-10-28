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

def helper(tmdp, patient_id, num_of_cf_trajectories):

    summaries = []
    A_real = tmdp.get_trajectory_actions(patient_id=patient_id)
    
    # Compute average entropy
    P = tmdp.get_P()
    mdp_entropy = []
    for a, trans in P.items():
        mdp_entropy.append(np.apply_along_axis(lambda x : entropy(x, base=2), axis=1, arr=trans))
    mean_mdp_entropy = np.mean(mdp_entropy)

    P_cf = tmdp.get_counterfactual_MDP(patient_id=patient_id, recompute=False)
    cf_entropy = []
    for (a, t), trans in P_cf.items():
        cf_entropy.append(np.apply_along_axis(lambda x : entropy(x, base=2), axis=1, arr=trans))
    mean_cf_entropy = np.mean(cf_entropy)

    horizon = np.max([x[1] for x in list(P_cf.keys())]) + 1

    # Compute average counterfactual outcome
    for k in range(horizon+1):
        summary = {}
        summary['mdp_entropy'] = mean_mdp_entropy
        summary['cf_entropy'] = mean_cf_entropy
        summary['id'] = int(patient_id)

        summary['k'] = k
        summary['horizon'] = int(horizon)
        pi, exp_reward = tmdp.maximize(patient_id=patient_id, k=k)
        summary['avg_cf_outcome'] = exp_reward

        # Compute number of unique counterfactual action sequences
        unique_counterfactuals = []
        for seed in range(1, num_of_cf_trajectories+1):
            _, _, a, _, _ = tmdp.sample_cf_trajectory(patient_id=patient_id, pi=pi, seed=seed)
            if a.tolist() not in unique_counterfactuals and a.tolist() != A_real:
                unique_counterfactuals.append(a.tolist())
        summary['num_of_explanations'] = len(unique_counterfactuals)
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
def therapy_experiment(num_of_cf_samples, cf_mdp_directory, n_jobs, outputs, num_of_cf_trajectories, min_horizon, data):

    tmdp = Therapy_MDP(data_filename=data, cf_mdp_directory=cf_mdp_directory, min_horizon=min_horizon)
    tmdp.initialize_MDP()
    patient_IDs = tmdp.get_valid_IDs()

    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(helper)(tmdp=tmdp, patient_id=patient_id, num_of_cf_trajectories=num_of_cf_trajectories) for patient_id in patient_IDs)
    final_results = [item for sublist in results for item in sublist]
    for summary in final_results:
        with open(outputs+'therapy_experiment_id_' + str(summary['id']) + '_k_' + str(summary['k']) + '.json', 'w') as f:
            json.dump(summary, f)

if __name__ == '__main__':
    therapy_experiment()
    # therapy_experiment(num_of_cf_samples=1000, cf_mdp_directory='outputs/cf_mdps/', n_jobs=1, outputs='outputs/',
    #             num_of_cf_trajectories=100, min_horizon=10, data='data/therapy/therapy.csv')