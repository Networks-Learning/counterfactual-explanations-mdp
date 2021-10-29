#!/bin/bash

num_of_trajectories=50
error_prob=0.05
horizon=20
num_of_actions=10
num_of_states=20
num_of_cf_samples=1000
uncertainty_param_list=(0 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
n_jobs=4
param_seeds=10
num_of_cf_trajectories=100

for j in {0..6}
do
    uncertainty_param=${uncertainty_param_list[$j]}
    python -m src.synth_experiment --num_of_trajectories=$num_of_trajectories --error_prob=$error_prob --horizon=$horizon --num_of_actions=$num_of_actions --num_of_states=$num_of_states --num_of_cf_samples=$num_of_cf_samples --uncertainty_param=$uncertainty_param --n_jobs=$n_jobs --param_seeds=$param_seeds --outputs=outputs/ --num_of_cf_trajectories=$num_of_cf_trajectories --cf_mdp_directory=outputs/cf_mdps/
done