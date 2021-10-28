import numpy as np
import pandas as pd
import pickle
from src.synth_mdp import Synth_MDP
# from synth_mdp import Synth_MDP
from joblib import Parallel, delayed
from copy import deepcopy, error
import click
from scipy.stats import entropy
import json
import time

def helper(smdp, trajectory_id, horizon, num_of_cf_trajectories, uncertainty_param, param_seed, error_prob):

    summaries = []
    A_real = smdp.get_trajectory_actions(trajectory_id=trajectory_id)
    
    # Compute average entropy
    P = smdp.get_P()
    mdp_entropy = []
    for a, trans in P.items():
        mdp_entropy.append(np.apply_along_axis(lambda x : entropy(x, base=2), axis=1, arr=trans))
    mean_mdp_entropy = np.mean(mdp_entropy)

    P_cf, _ = smdp.get_counterfactual_MDP(trajectory_id=trajectory_id, recompute=False)
    cf_entropy = []
    for (a, t), trans in P_cf.items():
        cf_entropy.append(np.apply_along_axis(lambda x : entropy(x, base=2), axis=1, arr=trans))
    mean_cf_entropy = np.mean(cf_entropy)

    # Compute average counterfactual outcome
    for k in range(horizon+1):
        summary = {}
        summary['mdp_entropy'] = mean_mdp_entropy
        summary['cf_entropy'] = mean_cf_entropy
        summary['alpha'] = uncertainty_param
        summary['id'] = trajectory_id
        summary['param_seed'] = param_seed
        summary['error_prob'] = error_prob
    
        summary['k'] = k
        start_time = time.time()
        pi, exp_reward = smdp.maximize(trajectory_id=trajectory_id, k=k)
        end_time = time.time()
        total_time = end_time-start_time

        summary['avg_cf_outcome'] = exp_reward
        summary['opt_time'] = total_time

        # Compute number of unique counterfactual action sequences
        unique_counterfactuals = []
        for seed in range(1, num_of_cf_trajectories+1):
            _, _, a, _, _ = smdp.sample_cf_trajectory(trajectory_id=trajectory_id, pi=pi, seed=seed)
            if a.tolist() not in unique_counterfactuals and a.tolist() != A_real:
                unique_counterfactuals.append(a.tolist())
        summary['num_of_explanations'] = len(unique_counterfactuals)
        summaries.append(summary)

    print('Done ID '+str(trajectory_id))
    return summaries

@click.command() # Comment the click commands for testing
@click.option('--num_of_states', type=int, required=True, help="Number of states")
@click.option('--num_of_actions', type=int, required=True, help="Number of actions")
@click.option('--num_of_cf_samples', type=int, required=True, help="Number of counterfactual samples")
@click.option('--error_prob', type=float, required=True, help="Prob of taking wrong action")
@click.option('--horizon', type=int, required=True, help="Horizon")
@click.option('--num_of_trajectories', type=int, required=True, help="Number of trajectories to sample")
@click.option('--uncertainty_param', type=float, required=True, help="Upper bound of uniform")
@click.option('--n_jobs', type=int, required=True, help="Number of parallel threads")
@click.option('--param_seeds', type=int, required=True, help="Number of different seed batches")
@click.option('--outputs', type=str, required=True, help="Output directory for counterfactual MDPs")
@click.option('--cf_mdp_directory', type=str, required=True, help="Output directory for counterfactual MDPs")
@click.option('--num_of_cf_trajectories', type=int, required=True, help="Number of counterfactual explanations")
def synth_experiment(num_of_states, num_of_actions, num_of_cf_samples, error_prob, horizon, num_of_trajectories,
                            cf_mdp_directory, uncertainty_param, n_jobs, param_seeds, outputs, num_of_cf_trajectories):

    for param_seed in range(1, param_seeds+1):
        smdp = Synth_MDP(num_of_actions=num_of_actions, num_of_states=num_of_states, uncertainty_param=uncertainty_param, param_seed=param_seed,
                    horizon=horizon, cf_mdp_directory=cf_mdp_directory, error_prob=error_prob)
        smdp.initialize_MDP(num_of_trajectories=num_of_trajectories)
        trajectory_IDs = smdp.get_IDs()

        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(helper)(smdp=smdp, trajectory_id=trajectory_id, horizon=horizon, num_of_cf_trajectories=num_of_cf_trajectories, uncertainty_param=uncertainty_param, param_seed=param_seed, error_prob=error_prob) for trajectory_id in trajectory_IDs)
        final_results = [item for sublist in results for item in sublist]
        for summary in final_results:
            with open(outputs+'synth_experiment_alpha_' + str(summary['alpha']) + '_paramseed_' + str(param_seed) +'_id_' + str(summary['id']) + '_k_' + str(summary['k']) + '_errorprob_' + str(error_prob) + '.json', 'w') as f:
                json.dump(summary, f)

if __name__ == '__main__':
    synth_experiment()
    # synth_experiment(num_of_actions=10, num_of_states=20, num_of_cf_samples=1000, error_prob=0.05, horizon=20, num_of_trajectories=100,
    #             cf_mdp_directory='outputs/cf_mdps/', uncertainty_param=0.0, n_jobs=1, param_seeds=10, outputs='outputs/',
    #             num_of_cf_trajectories=100)