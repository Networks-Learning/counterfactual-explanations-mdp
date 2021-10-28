#!/bin/bash

min_horizon=10
num_of_cf_samples=1000
n_jobs=4
unobserved_reward=inf       # Use this option for the experiments in the main (Section 6)
# unobserved_reward=normal  # Use this option for the comparison with baselines (Appendix D)

python -m src.therapy_compute_cf_mdps --min_horizon=$min_horizon --num_of_cf_samples=$num_of_cf_samples --n_jobs=$n_jobs --data=data/therapy/therapy.csv --outputs=outputs/cf_mdps/ --unobserved_reward=$unobserved_reward