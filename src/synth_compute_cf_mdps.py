import numpy as np
import pandas as pd
import pickle
from src.synth_mdp import Synth_MDP
from joblib import Parallel, delayed
from copy import deepcopy
import click

@click.command() # Comment the click commands for testing
@click.option('--num_of_states', type=int, required=True, help="Number of states")
@click.option('--num_of_actions', type=int, required=True, help="Number of actions")
@click.option('--num_of_cf_samples', type=int, required=True, help="Number of counterfactual samples")
@click.option('--error_prob', type=float, required=True, help="Prob of taking wrong action")
@click.option('--horizon', type=int, required=True, help="Horizon")
@click.option('--num_of_trajectories', type=int, required=True, help="Number of trajectories to sample")
@click.option('--uncertainty_param', type=float, required=True, help="Upper bound of uniform")
@click.option('--n_jobs', type=int, required=True, help="Number of parallel threads")
@click.option('--param_seeds', type=int, required=True, help="Number of different seeds for the MDP parameters")
@click.option('--outputs', type=str, required=True, help="Output directory for counterfactual MDPs")
def synth_compute_cf_mdps(num_of_states, num_of_actions, num_of_cf_samples, error_prob, horizon, num_of_trajectories, uncertainty_param, n_jobs, param_seeds, outputs):

    for seed in range(1, param_seeds+1):
        smdp = Synth_MDP(num_of_actions=num_of_actions, num_of_states=num_of_states, uncertainty_param=uncertainty_param, param_seed=seed,
                    horizon=horizon, cf_mdp_directory=outputs, error_prob=error_prob)
        smdp.initialize_MDP(num_of_trajectories=num_of_trajectories)

        trajectory_IDs = smdp.get_IDs()
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(smdp.get_counterfactual_MDP)(trajectory_id=trajectory_id, num_of_cf_samples=num_of_cf_samples, recompute=True, verbose=True) for trajectory_id in trajectory_IDs)
        
        times = [x[1] for x in results]
        with open(outputs+'cf_mdp_time.txt', 'w') as f:
            f.write('Average CF MDP computation time: ' + str(np.mean(times)) + '\n')

if __name__ == '__main__':
    synth_compute_cf_mdps()