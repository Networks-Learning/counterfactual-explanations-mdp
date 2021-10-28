#!/bin/bash

min_horizon=10
num_of_cf_samples=1000
n_jobs=4
num_of_cf_trajectories=1000

python -m src.therapy_experiment --min_horizon=$min_horizon --num_of_cf_samples=$num_of_cf_samples --n_jobs=$n_jobs --outputs=outputs/ --data=data/therapy/therapy.csv --num_of_cf_trajectories=$num_of_cf_trajectories --cf_mdp_directory=outputs/cf_mdps/
