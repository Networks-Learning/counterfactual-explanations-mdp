import numpy as np
import pandas as pd
import pickle
import os.path

class Therapy_MDP():
    
    def __init__(self, data_filename, cf_mdp_directory, min_horizon):

        self.states = {
            0 : "0-4",
            1 : "5-9",
            2 : "10-14",
            3 : "15-19",
            4 : "20-27",
            5 : "absorbing"
        }

        self.data_filename = data_filename
        self.cf_mdp_directory = cf_mdp_directory
        self.min_horizon = min_horizon

    def initialize_MDP(self, unobserved_reward='inf'):

        self.raw_df = self._read_csv_file()
        self.trajectories = self._extract_trajectories()
        self.unobserved_reward = unobserved_reward
        self.P, self.R = self._fit_mdp_parameters()

        return
    
    def _read_csv_file(self):
        
        df = pd.read_csv(self.data_filename, delimiter=';', header=0)
        df = df.drop(columns=['Score'])
        
        severity_column = []
        for _, row in df.iterrows():
            row_sum = 0
            row_valid_answers = 0

            # Scale the severity score based on the numbered of answered questions
            for i in range(1,10):
                if (row['FRAGE0'+str(i)+'_1'] != '') and (row['FRAGE0'+str(i)+'_1'] != 'Z') and (row['FRAGE0'+str(i)+'_1'] != ' '):
                    row_sum += int(row['FRAGE0'+str(i)+'_1'])
                    row_valid_answers += 1
            
            score = np.rint(9/row_valid_answers * row_sum)
            
            if score >=0 and score <= 4:
                severity = 0
            elif score >=5 and score <= 9:
                severity = 1
            elif score >= 10 and score <= 14:
                severity = 2
            elif score >= 15 and score <= 19:
                severity = 3
            elif score >= 20 and score <= 27:
                severity = 4
            else:
                print('Out of bounds')
            
            severity_column.append(severity)

        df['Severity'] = severity_column
        df = df.drop(columns=['FRAGE0'+str(i)+'_1' for i in range(1,10)])
        return df

    def _extract_trajectories(self):

        patient_IDs = self.raw_df.patnr.unique()
        trajectories = {}
        for patient_id in patient_IDs:
            reward = 0
            patient_df = self.raw_df[self.raw_df['patnr'] == patient_id]
            trajectories[patient_id] = {'states' : [], 'actions' : []}
            for _, row in patient_df.iterrows():
                trajectories[patient_id]['states'].append(int(row['Severity']))
                trajectories[patient_id]['actions'].append(int(row['Action']))
                reward += 5 - trajectories[patient_id]['states'][-1]
            
            # If the trajectory finished without an Abschluss action,
            # add a dummy last meeting with the same severity state.
            if trajectories[patient_id]['actions'][-1] != 10:
                trajectories[patient_id]['actions'].append(10)
                trajectories[patient_id]['states'].append(trajectories[patient_id]['states'][-1])
                reward += 5 - trajectories[patient_id]['states'][-1]
            
            trajectories[patient_id]['reward'] = reward
            
            # If the trajectory's horizon is less than a threshold, discard it
            if len(trajectories[patient_id]['states'])<self.min_horizon:
                del trajectories[patient_id]

        return trajectories

    def _fit_mdp_parameters(self, num_of_dirichlet_samples=100000):

        transitions = {}
        for _, trajectory in self.trajectories.items():
            states = trajectory['states']
            actions = trajectory['actions']
            for ind, action in enumerate(actions[:-1]):
                previous_state = states[ind]
                next_state = states[ind+1]
                if (action, previous_state) not in transitions:
                    transitions[action, previous_state] = []
                transitions[action, previous_state].append(next_state)
        
        R = {}
        P = {}
        hyperprior = {}
        if self.unobserved_reward == 'normal':
            fill_value = 0
        elif self.unobserved_reward == 'inf':
            fill_value = -np.inf
        
        for (a, s), s_p_arr in transitions.items():
            
            if a not in P:
                P[a] = np.zeros((6,6))
                hyperprior[a] = np.full((6,6),fill_value=0.01)
                for s_center in range(6):
                    # hyperprior[a][s_center,s_center] = 1    
                    for s_neighbor in range(6):
                        if s_neighbor == s_center-1 or s_neighbor == s_center+1 or s_neighbor == s_center:
                            hyperprior[a][s_center,s_neighbor] = 1

                R[a] = np.full(6, fill_value) # Action-state pairs that never occured give -infinite reward by default
            
            for s_p in s_p_arr:
                hyperprior[a][s, s_p] += 1
                # P[a][s, s_p] += 1
            
            R[a][s] = 5 - s     # Reward 1 for state 4 (high severity), reward 5 for state 0 (low severity)

        # Here we define what happens with the last action (Abschluss -- 10)
        R[10] = np.full(6, fill_value)
        for s in range(5):
            R[10][s] = 5 - s 
        
        for s in range(6):
            P[10] = np.zeros((6,6))
            hyperprior[10] = np.full((6,6),fill_value=0.01)
        
        for a in hyperprior:
            for s in range(6):
                rng = np.random.default_rng(seed=1)
                dir_samples = rng.dirichlet(alpha=hyperprior[a][s,:], size=num_of_dirichlet_samples)
                P[a][s,:] = np.mean(dir_samples, axis=0)
                assert np.count_nonzero(P[a][s,:])!=0, 'Something went wrong with the priors'
        
        if self.unobserved_reward == 'normal':
            for a in P:
                for s in range(6):
                    R[a][s] = 5 - s
        
        return P, R


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
        for seed in range(num_of_samples):
            rng = np.random.default_rng(seed+1)
            gumbels.append(topdown(trans_probabilities, s_p_real, rng))

        # Sanity check
        for gum in gumbels:
            temp = gum + np.log(trans_probabilities)
            assert np.argmax(temp)==s_p_real, "Sampled gumbels don't match with realized argmax"
        
        return gumbels

    def get_counterfactual_MDP(self, patient_id, num_of_cf_samples=1000, verbose=False, recompute=False):
        
        if verbose:
            print('Samples: ' + str(num_of_cf_samples) + ', ID: ' + str(patient_id))

        pickle_name = self.cf_mdp_directory + 'therapy_cf_mdp_reward_' +str(self.unobserved_reward) + '_id_' + str(patient_id) + '_samples_' + str(num_of_cf_samples) + '.pkl'

        try:
            if recompute:
                raise Exception
            else:
                with open(pickle_name, 'rb') as f:
                    P_cf = pickle.load(f)
        except:
            
            if not recompute:
                print('Had to recompute')
            
            states = self.trajectories[patient_id]['states']
            actions = self.trajectories[patient_id]['actions']

            P_cf = {}
            for t in range(len(states)-1):
                s_real, s_p_real = states[t], states[t+1]
                a_real = actions[t]
                
                # Sample from the noise posterior
                gumbels_set = self.__sample_gumbels(self.P[a_real][s_real], s_p_real, num_of_cf_samples)
                
                for a in self.P:
                    P_cf[a,t] = np.zeros((6,6))
                    for s in range(5):
                        for gumbels in gumbels_set:
                            P_cf[a,t][s,np.argmax(gumbels + np.log(self.P[a][s]))] += 1 # Set according to the SCM
                    P_cf[a,t][5,5] = 1 # Set the last state as absorbing
                    P_cf[a,t] = P_cf[a,t]/P_cf[a,t].sum(axis=1, keepdims=1)
                
                P_cf[10,t] = np.zeros((6,6))
                for s in range(6):
                    P_cf[10,t][s, 5] = 1             # Action 10 (Abschluss) deterministically leads to the absorbing state
            
            t = len(states)-1   # In the last step, we cannot compute a counterfactual distribution
            for a in self.P:
                P_cf[a, t] = self.P[a].copy()
            
            P_cf[10, t] = np.zeros((6,6))
            for s in range(6):
                P_cf[10,t][s,5] = 1

            with open(pickle_name, 'wb') as f:
                pickle.dump(P_cf, f)

        return P_cf

    
    def maximize(self, patient_id, k):
        
        P_cf = self.get_counterfactual_MDP(patient_id=patient_id, recompute=False)
        T = np.max([x[1] for x in list(P_cf.keys())])+1     # Horizon = last time step + 1 
        R = self.R
        s_0 = self.trajectories[patient_id]['states'][0]
        A_real = self.trajectories[patient_id]['actions']

        h_fun = np.zeros((6, T+1, k+1))
        pi = np.zeros((6, k+1, T+1), dtype=int)
        pi[:, 0, T] = np.zeros(6, dtype=int)    # Set action 0 to first time step.

        # If there is one step remaining, you just play 10 (Abschluss), even if there are changes left.
        for c in range(0, k+1):
            for s in range(6):
                h_fun[s, 1, c] = R[10][s]
                pi[s, k-c, T-1] = 10
        
        # If there are no changes left (c=0), just play the observed action
        for r in range(2, T):
            for s in range(6):
                h_fun[s, r, 0] = R[A_real[T-r]][s]
                for s_p in range(6):
                    if P_cf[A_real[T-r], T-r][s, s_p] != 0: # Make sure to avoid 0 * inf multiplication
                        h_fun[s, r, 0] += P_cf[A_real[T-r], T-r][s, s_p] *  h_fun[s_p, r-1, 0]
                pi[s, k, T-r] = A_real[T-r]
        
        # For t=1,...,T-2 do recursive computations
        for r in range(2, T):
            for c in range(1, k+1):
                for s in range(6):
                    max_val = -np.inf
                    for a in self.P:
                        if a != 0 and a != 10:
                            val = R[a][s]
                            if  a != A_real[T-r]:
                                for s_p in range(6):
                                    if P_cf[a, T-r][s, s_p] != 0:
                                        val += P_cf[a, T-r][s, s_p] * h_fun[s_p, r-1, c-1]
                            elif a == A_real[T-r]:
                                for s_p in range(6):
                                    if P_cf[a, T-r][s, s_p] != 0:
                                        val += P_cf[a, T-r][s, s_p] * h_fun[s_p, r-1, c]
                            
                            # Pick best action
                            if val > max_val:
                                max_val = val
                                best_act = a

                    h_fun[s, r, c] = max_val
                    if max_val == -np.inf:
                        pi[s, k-c, T-r] = -1
                    else:
                        pi[s, k-c, T-r] = best_act
        
        # During the first step, in any case, play action 0
        for c in range(0, k+1):
            for s in range(6):
                h_fun[s, T, c] = R[0][s]
                for s_p in range(6):
                    if P_cf[0, 0][s, s_p] != 0:
                        h_fun[s, T, c] += P_cf[0, 0][s, s_p] * h_fun[s_p, T-1, c]
                pi[s, k-c, 0] = 0

        return pi, h_fun[s_0, T, k]

    def sample_cf_trajectory(self, patient_id, pi, k=3, seed=1):
        
        P_cf = self.get_counterfactual_MDP(patient_id=patient_id, recompute=False)
        T = np.max([x[1] for x in list(P_cf.keys())]) + 1
        R = self.R
        A_real = self.trajectories[patient_id]['actions']
        rng = np.random.default_rng(seed=seed)

        reward = 0
        s = np.zeros(T, dtype=int)
        s[0] = self.trajectories[patient_id]['states'][0]   # Initial state the same
        l = np.zeros(T, dtype=int)
        l[0] = 0    # Start with 0 changes
        a = np.zeros(T, dtype=int)
        prob = 1    # Probability to observe the sampled trajectory
        
        for t in range(T):
            if type(pi) == str:
                if t == 0:
                    a[t] = 0
                elif t == T-1:
                    a[t] = 10
                else:
                    valid_actions = []
                    for act in list(R.keys()):
                        if act != 0 and act != 10 and R[act][s[t]] != -np.inf:
                            valid_actions.append(act)
                    if valid_actions == [] or l[t]==k:
                        a[t] = A_real[t]
                    else:
                        if pi == 'random':
                            a[t] = rng.choice(valid_actions)
                        elif pi == 'greedy':
                            future_rewards = np.zeros(len(valid_actions), dtype=float)
                            for ind, act in enumerate(valid_actions):
                                future_rewards[ind] = np.dot(R[0], P_cf[act, t][s[t], :])

                            max_val = -np.inf
                            for ind, act in enumerate(valid_actions):
                                if future_rewards[ind] > max_val:
                                    max_val = future_rewards[ind]
                                    best_act = act
                            if max_val == -np.inf:
                                a[t] = A_real[t]
                            else:
                                a[t] = best_act
                            
                        elif pi == 'randomized':
                            
                            future_rewards = np.zeros(len(valid_actions), dtype=float)
                            for ind, act in enumerate(valid_actions):
                                future_rewards[ind] = np.dot(R[0], P_cf[act, t][s[t], :])

                            max_val = -np.inf
                            for ind, act in enumerate(valid_actions):
                                if future_rewards[ind] > max_val:
                                    max_val = future_rewards[ind]
                                    best_act = act
                            
                            if max_val == -np.inf or rng.binomial(n=1, p=0.5)==1:
                                a[t] = A_real[t]
                            else:
                                a[t] = best_act

                            
            else:
                a[t] = pi[s[t], l[t], t]    # Pick actions according to the given policy
            
            reward += R[a[t]][s[t]]     # Get the immediate reward
            if t != T-1:
                s[t+1] = rng.choice(a=range(6), size=1, p=P_cf[a[t], t][s[t],:])    # Sample the next state
                prob *= P_cf[a[t], t][s[t],s[t+1]]  # Adjust the probability of the trajectory
                if a[t] != A_real[t]:
                    l[t+1] = l[t] + 1   # Adjust the number of changes so far
                else:
                    l[t+1] = l[t]
    
        return s, l, a, reward, prob

    def get_valid_IDs(self, min_length=0):
        
        valid_IDs = []
        patient_IDs = list(self.trajectories.keys())
        for patient_ID in patient_IDs:
            if len(self.trajectories[patient_ID]['states']) >= min_length:
                valid_IDs.append(patient_ID)

        return valid_IDs

    def get_trajectory_actions(self, patient_id):
        actions = self.trajectories[patient_id]['actions']
        return actions
    
    def get_trajectory_states(self, patient_id):
        states = self.trajectories[patient_id]['states']
        return states

    def get_trajectory_reward(self, patient_id):
        reward = self.trajectories[patient_id]['reward']
        return reward

    def get_P(self):
        return self.P

# These lines are for testing
# test = Therapy_MDP(data_filename='data/therapy/therapy.csv', cf_mdp_directory='outputs/cf_mdps/', min_horizon=10)
# test.initialize_MDP(unobserved_reward='normal')
# P_cf = test.get_counterfactual_MDP(patient_id=7, recompute=False, num_of_cf_samples=1000, verbose=True)
# pi, exp_reward = test.maximize(patient_id=7, k=2)
# tau = test.sample_cf_trajectory(patient_id=7, pi='randomized', k=3, seed=2)
# print('END')