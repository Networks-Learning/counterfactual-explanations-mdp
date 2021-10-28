import numpy as np
import pandas as pd
import pickle
from src.therapy_mdp import Therapy_MDP
# from therapy_mdp import Therapy_MDP
from joblib import Parallel, delayed
from copy import deepcopy
import click
    
@click.command() # Comment the click commands for testing
@click.option('--num_of_cf_samples', type=int, required=True, help="Number of counterfactual samples")
@click.option('--min_horizon', type=int, required=True, help="Minimum horizon")
@click.option('--n_jobs', type=int, required=True, help="Number of parallel threads")
@click.option('--outputs', type=str, required=True, help="Output directory for counterfactual MDPs")
@click.option('--data', type=str, required=True, help="File containing the therapy data")
@click.option('--unobserved_reward', type=click.Choice(['inf', 'normal']), required=True, help="What to do with rewards of unobserved action-states")
def therapy_compute_cf_mdps(num_of_cf_samples, n_jobs, min_horizon, outputs, data, unobserved_reward):

    tmdp = Therapy_MDP(data_filename=data, cf_mdp_directory=outputs, min_horizon=min_horizon)
    tmdp.initialize_MDP(unobserved_reward=unobserved_reward)
    
    patient_IDs = tmdp.get_valid_IDs()
    _ = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(tmdp.get_counterfactual_MDP)(patient_id=patient_id, num_of_cf_samples=num_of_cf_samples, recompute=True, verbose=True) for patient_id in patient_IDs)

if __name__ == '__main__':
    therapy_compute_cf_mdps()
    # therapy_compute_cf_mdps(num_of_cf_samples=1000, n_jobs=1, min_horizon=10, outputs='outputs/cf_mdps/',
    #                         data='data/therapy/therapy.csv', unobserved_reward=0.0)