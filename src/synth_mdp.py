import numpy as np
import pandas as pd
import pickle
import os.path
import time
class Synth_MDP():
    
    def __init__(self, num_of_actions, num_of_states, uncertainty_param, param_seed, horizon, cf_mdp_directory, error_prob):

        self.states = range(num_of_states)
        self.actions = range(num_of_actions)
        self.param_seed = param_seed
        self.horizon = horizon
        self.uncertainty_param = uncertainty_param
        self.cf_mdp_directory = cf_mdp_directory
        self.error_prob = error_prob

    def initialize_MDP(self, num_of_trajectories):

        self.P, self.R = self._set_mdp_parameters()
        self.trajectories = self._sample_trajectories(num_of_trajectories)

        return
    
    def _set_mdp_parameters(self):
        
        R = {}
        for a in self.actions:
            R[a] = np.array([s for s in self.states])
        self.R = R

        P = {}
        rng = np.random.default_rng(seed=self.param_seed)
        for a in self.actions:
            P[a] = np.zeros((len(self.states), len(self.states)))
            for s in self.states:
                weights = rng.uniform(low=0, high=self.uncertainty_param, size=len(self.states))
                weights[rng.choice(self.states)] = 1
                # weights = rng.normal(loc=1.0, scale=self.uncertainty_param, size=len(self.states))
                P[a][s, :] = weights
                # if np.count_nonzero(P[a][s, :])==0:
                #     P[a][s, :] = np.ones(len(self.states))
        for a in P:
            P[a] = P[a]/P[a].sum(axis=1, keepdims=1)
        
        return P, R
    
    def _mdp_optimal_policy(self):
        
        P = self.P
        R = self.R
        T = self.horizon

        h_fun = np.zeros((len(self.states), T+1))
        pi = np.zeros((len(self.states), T+1), dtype=int)
        
        for r in range(1, T+1):
            for s in self.states:
                max_val = -np.inf
                for a in self.actions:
                    val = R[a][s]
                    for s_p in self.states:
                        val += P[a][s, s_p] * h_fun[s_p, r-1]

                    if val > max_val:
                        max_val = val
                        best_act = a

                h_fun[s, r] = max_val
                pi[s, T-r] = best_act

        return pi

    def _sample_trajectories(self, num_of_trajectories):

        trajectory_IDs = range(num_of_trajectories)
        P = self.P
        R = self.R
        
        pi = self._mdp_optimal_policy()

        trajectories = {}
        for trajectory_id in trajectory_IDs:
            rng = np.random.default_rng(seed=trajectory_id)
            reward = 0
            trajectories[trajectory_id] = {'states' : [], 'actions' : []}
            trajectories[trajectory_id]['states'].append(rng.choice(self.states))
            for t in range(self.horizon):
                s = trajectories[trajectory_id]['states'][t]
                a_opt = pi[s, t]
                a = a_opt
                if rng.binomial(1, self.error_prob)==1:
                    while a == a_opt:
                        a = rng.choice(self.actions)
                    
                trajectories[trajectory_id]['actions'].append(a)
                reward += R[a][s]
                if t != self.horizon-1:
                    s_p = rng.choice(a=self.states, p=P[a][s,:])    # Sample the next stat
                    trajectories[trajectory_id]['states'].append(s_p)
            
            trajectories[trajectory_id]['reward'] = reward
        
        return trajectories


    def __sample_gumbels(self, trans_probabilities, s_p_real, num_of_samples):
        
        #############################################
        # This part is adapted from https://cmaddis.github.io/gumbel-machinery
        
        def truncated_gumbel(alpha, truncation, rng):
            gumbel = rng.gumbel() + np.log(alpha)
            return -np.log(np.exp(-gumbel) + np.exp(-truncation))
        
        def topdown(alphas, k, rng):
            topgumbel = rng.gumbel() + np.log(sum(alphas))
            gumbels = []
            for i in range(len(alphas)):
                if i == k:
                    gumbel = topgumbel - np.log(trans_probabilities[i])
                elif trans_probabilities[i]!=0:
                    gumbel = truncated_gumbel(alphas[i], topgumbel, rng) - np.log(trans_probabilities[i])
                else:
                    gumbel = rng.gumbel() # When the probability is zero, sample an unconstrained Gumbel

                gumbels.append(gumbel)
            return gumbels
        #############################################

        gumbels = []
        for seed in range(1, num_of_samples+1):
            rng = np.random.default_rng(seed)
            gumbels.append(topdown(trans_probabilities, s_p_real, rng))

        # Sanity check
        for gum in gumbels:
            temp = gum + np.log(trans_probabilities)
            assert np.argmax(temp)==s_p_real, "Sampled gumbels don't match with realized argmax"
        
        return gumbels

    def get_counterfactual_MDP(self, trajectory_id, num_of_cf_samples=1000, verbose=False, recompute=False):
        
        if verbose:
            print('Uncertainty: ' + str(self.uncertainty_param) +', Param.Seed: ' + str(self.param_seed) + ', Samples: ' + str(num_of_cf_samples) \
                    + ', ID: ' + str(trajectory_id))
        
        pickle_name = self.cf_mdp_directory + 'synth_cf_mdp_id_' + str(trajectory_id) + '_uncer_' + str(self.uncertainty_param) \
                        + '_paramseed_' + str(self.param_seed) + '_samples_' + str(num_of_cf_samples) + '_errorprob_' + str(self.error_prob) + '.pkl'

        total_time=0

        try:
            if recompute:
                raise Exception
            else:
                with open(pickle_name, 'rb') as f:
                    P_cf = pickle.load(f)
        except:
            
            if not recompute:
                print('Had to recompute')
            
            start_time = time.time()

            states = self.trajectories[trajectory_id]['states']
            actions = self.trajectories[trajectory_id]['actions']

            P_cf = {}
            for t in range(self.horizon-1):
                s_real, s_p_real = states[t], states[t+1]
                a_real = actions[t]
                
                # Sample from the noise posterior
                gumbels_set = self.__sample_gumbels(self.P[a_real][s_real], s_p_real, num_of_cf_samples)
                
                for a in self.actions:
                    P_cf[a,t] = np.zeros((len(self.states),len(self.states)))
                    for s in self.states:
                        for gumbels in gumbels_set:
                            P_cf[a,t][s,np.argmax(gumbels + np.log(self.P[a][s]))] += 1 # Set according to the SCM
                    P_cf[a,t] = P_cf[a,t]/P_cf[a,t].sum(axis=1, keepdims=1)
            
            t = self.horizon-1   # In the last step, we cannot compute a counterfactual distribution
            for a in self.P:
                P_cf[a, t] = self.P[a].copy()

            end_time = time.time()
            total_time = end_time - start_time

            with open(pickle_name, 'wb') as f:
                pickle.dump(P_cf, f)

        return P_cf, total_time

    def maximize(self, trajectory_id, k):
        
        P_cf, _ = self.get_counterfactual_MDP(trajectory_id=trajectory_id, recompute=False)
        T = np.max([x[1] for x in list(P_cf.keys())]) + 1   # Horizon = last time step + 1 
        R = self.R
        s_0 = self.trajectories[trajectory_id]['states'][0]
        A_real = self.trajectories[trajectory_id]['actions']

        h_fun = np.zeros((len(self.states), T+1, k+1))
        pi = np.zeros((len(self.states), k+1, T+1), dtype=int)

        # If there are no changes left (c=0), just play the observed action
        for r in range(1, T+1):
            for s in self.states:
                h_fun[s, r, 0] = R[A_real[T-r]][s]
                for s_p in self.states:
                    h_fun[s, r, 0] += P_cf[A_real[T-r], T-r][s, s_p] * h_fun[s_p, r-1, 0]
                pi[s, k, T-r] = A_real[T-r]
        
        # For t=1,...,T-2 do recursive computations
        for r in range(1, T+1):
            for c in range(1, k+1):
                for s in self.states:
                    max_val = -np.inf
                    for a in self.actions:
                        val = R[a][s]
                        if a != A_real[T-r]:
                            for s_p in self.states:
                                val += P_cf[a, T-r][s, s_p] * h_fun[s_p, r-1, c-1]
                        elif a == A_real[T-r]:
                            for s_p in self.states:
                                val += P_cf[a, T-r][s, s_p] * h_fun[s_p, r-1, c]

                        if val > max_val:
                            max_val = val
                            best_act = a

                    h_fun[s, r, c] = max_val
                    pi[s, k-c, T-r] = best_act

        return pi, h_fun[s_0, T, k]
    

    def sample_cf_trajectory(self, trajectory_id, pi, seed=1):
        
        P_cf, _ = self.get_counterfactual_MDP(trajectory_id=trajectory_id, recompute=False)
        T = np.max([x[1] for x in list(P_cf.keys())]) + 1
        R = self.R
        A_real = self.trajectories[trajectory_id]['actions']
        rng = np.random.default_rng(seed=seed)

        reward = 0
        s = np.zeros(T, dtype=int)
        s[0] = self.trajectories[trajectory_id]['states'][0]   # Initial state the same
        l = np.zeros(T, dtype=int)
        l[0] = 0    # Start with 0 changes
        a = np.zeros(T, dtype=int)
        prob = 1    # Probability to observe the sampled trajectory
        
        for t in range(T):
            a[t] = pi[s[t], l[t], t]    # Pick actions according to the given policy
            reward += R[a[t]][s[t]]     # Get the immediate reward
            if t != T-1:
                s[t+1] = rng.choice(a=self.states, p=P_cf[a[t], t][s[t],:])    # Sample the next state
                prob *= P_cf[a[t], t][s[t],s[t+1]]  # Adjust the probability of the trajectory
                if a[t] != A_real[t]:
                    l[t+1] = l[t] + 1   # Adjust the number of changes so far
                else:
                    l[t+1] = l[t]
    
        return s, l, a, reward, prob
    
    def get_initial_state(self, trajectory_id):
        return self.trajectories[trajectory_id]['states'][0]
    
    def get_IDs(self):
        valid_IDs = list(self.trajectories.keys())
        return valid_IDs
    
    def get_trajectory_actions(self, trajectory_id):
        actions = self.trajectories[trajectory_id]['actions']
        return actions

    def get_P(self):
        return self.P

# These lines are for testing
# test = Synth_MDP(num_of_actions=10, num_of_states=7, uncertainty_param=0.1, param_seed=1,
#                 horizon=20, cf_mdp_directory='data/cf_mdps/', num_of_cf_samples=1000)
# test.initialize_MDP(num_of_trajectories=20, error_prob=0.05)
# P_cf = test._get_counterfactual_MDP(trajectory_id=7, recompute=False)
# pi, exp_reward = test.maximize(trajectory_id=7, k=2)
# tau = test.sample_cf_trajectory(trajectory_id=7, pi=pi, seed=5)
# print('END')