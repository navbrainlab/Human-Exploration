
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from tqdm import tqdm

from scipy.special import softmax
import random

###
from scipy.special import softmax 
from scipy.stats import halfnorm, uniform
from copy import deepcopy
# self-defined visualization
#from utils.viz import viz

#viz.get_style()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------- #
#              Auxiliary              #
# ----------------------------------- #

to_idx = lambda x: np.sum([3**j*(int(s)-1) 
            for j, s in enumerate(reversed(str(x)))])

to_phi = lambda x, nF: np.hstack([np.eye(nF
            )[int(s)-1] for s in str(x)])

# ----------------------------------- #
#                Models               #
# ----------------------------------- #

class simpleBuffer:

    def __init__(self):
        self.keys = ['s', 'a', 'r']
        self.reset()

    def push(self, m_dict):
        '''Add a sample trajectory'''
        for k in m_dict.keys():
            self.m[k] = m_dict[k]

    def sample(self, *args):
        '''Sample a trajectory'''
        lst = [self.m[k] for k in args]
        if len(lst) == 1: return lst[0]
        else: return lst 

    def reset(self):
        '''Empty the cached trajectory'''
        self.m = {k: 0 for k in self.keys}

class baseAgent:
    '''The base agent
    
    This is an abstract class of an agent.
    Each agent should contain at least 3
    static propoerty:

        name: what the agent is referred to
        pname: the name of tis parameter
        pval: the mean value of each parameters
            published in Niv's (2015) paper

    The agent has three modules of the function
        
        init: some initialization, embeddings, etc
        policy: how does the agent generate an action.
            Also known as forward/predict process
        learn: how does the agent adjust his/her knowledge
            of the environment. Can be understood as 
            backward/train process.
    '''
    name  = 'base'
    pname = []
    pval  = []

    def __init__(self, nD, nF, params):
        self.nD = nD
        self.nF = nF
        self.nS = nD**nF
        self._load_params(params)
        self._init_buffer()

    def _init_buffer(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        raise NotImplementedError

    def policy(self, stims):
        n = len(stims)
        return np.ones([n])/n

    def learn(self):
        raise NotImplementedError
    
class RandomPolicy(baseAgent):
    def __init__(self, nD, nF):
        self.ns = nD*nF
    
    def policy(self, obs):
        Q_vals = np.random.randn(obs.shape[0])
        return softmax(Q_vals)

class Epsilon_RL:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize a Naïve RL Agent using Q-learning.
        
        Args:
            state_size (int): Number of possible states.
            action_size (int): Number of possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Probability of random action (exploration).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        """
        Choose an action using an ε-greedy policy.
        
        Args:
            state (int): Current state.
        
        Returns:
            int: Action to take.
        """
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.randint(0, self.action_size - 1)
        else:  # Exploitation (choose best known action)
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.
        
        Args:
            state (int): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Next state after taking the action.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def get_q_table(self):
        """Return the current Q-table."""
        return self.q_table
    
class naiveRL(baseAgent):
    '''The naive RL'''
    name  = 'naive RL'
    pname = ['η', 'β']
    pval  = [1, 5.55, 0.1]
    
    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)
    
    def _load_params(self, params):
        self.eta  = params[0]
        self.beta = params[1]
        self.epsilon = params[2]
    
    # --------- decision --------- #

    def policy(self, stims, trial):

        # if trial==0:
        #     return np.random.permutation(range(9))
        # else:
            #     # Softmax(βV(s_i))
        P_s = softmax(self.beta*stims)  # probability distribution
        action = []
        print(stims)
        #print(P_s)
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            action = np.random.choice(9, size=9, replace=False)
            #print(action)
            #print('stochastic')
            return action
        else:  # Exploitation (choose best known action)
            #print('deterministic')
            # return np.argsort(P_s)[::-1]
            random_noise = np.random.rand(len(P_s)) * 1e-16
            P_s = P_s+random_noise
            action = np.argsort(P_s + random_noise)[::-1]
            return action
        
    
    # --------- learning --------- #

    def learn(self, action, r, Q_table):
        Q_table = self.update_V(action, r, Q_table)
        return Q_table

    def update_V(self, action, r, Q_table):

        # update: V(s_chosen) = V(s_chosen) + η(r-V(s_chosen))
        coefficient = [0.5, 0, -0.5]  ## can be free parameters
        for i in range(action.shape[0]):
            j = int(i/3)
            rpe = r/9 - Q_table[action[i]]
            #print(action[i])
            #print(j)
            #print(Q_table[action[(0+j*3):(3+j*3)]])
            Q_table[action[i]] += coefficient[j]*self.eta*rpe 
            
        return Q_table

class naiveRL_decay(baseAgent):
    '''The naive RL'''
    name  = 'naive RL'
    pname = ['η', 'β']
    pval  = [.431, 5.55, 0.1]
    
    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)

        
    def _load_params(self, params):
        self.eta  = params[0]
        self.beta = params[1]
        self.epsilon = params[2]

    # --------- decision --------- #

    def policy(self, stims):

        
        # Softmax(βV(s_i))
        P_s = softmax(self.beta*stims)  # probability distribution
        action = []
        #print(P_s)
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            action = np.random.choice(9, size=9, replace=False)
            #print(action)
            #print('stochastic')
            return action
        else:  # Exploitation (choose best known action)
            #print('deterministic')
            return np.argsort(P_s)[::-1]
        
    
    # --------- learning --------- #

    def learn(self, action, r, Q_table):
        Q_table = self.update_V(action, r, Q_table)
        return Q_table

    def update_V(self, action, r, Q_table):

        # update: V(s_chosen) = V(s_chosen) + η(r-V(s_chosen))
        coefficient = [0.6, 0.3, 0.1]  ## can be free parameters
        #print(Q_table)
        #exit()
        for i in range(action.shape[0]):
            j = int(i/3)
            rpe = coefficient[j]*r - Q_table[action[i]]
            #print(action[i])
            #print(j)
            #print(Q_table[action[(0+j*3):(3+j*3)]])
            Q_table[action[i]] += self.eta*rpe 
        
        all_index = np.arange(0, Q_table.shape[0])
        decay_index = np.setdiff1d(all_index, action)
        #print(decay_index)
        #exit()
        for i in range(decay_index.shape[0]):
            Q_table[decay_index[i]] = Q_table[decay_index[i]]*0.9  # decay rate  
        return Q_table

class fRL(baseAgent):
    '''The feature RL'''
    name  = 'feature RL'
    pname = ['η', 'β']
    pval  = [.43, 14.73]

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)
        self._init_W()

    def _load_params(self, params):
        self.eta  = params[0]
        self.beta = params[1]

    def _init_W(self):
        self.W = np.ones([self.nD*self.nF])*(1/(self.nD*self.nF))
        self.W = self.W.reshape(self.nD,self.nF)
        self.Q_buffer = []
        self.Q_index = []
    # --------- decision --------- #

    def policy(self, input_data):

        # Softmax(βV(s_i)): V(s_i) = ∑_fW(f)
        Q_values = []
        #print(input_data)
        for i in range(input_data.shape[0]):
            q_val = 0
            for j in range(input_data.shape[1]):
                # print(input_data[i])
                q_val += (self.W[j][input_data[i][j]-1]) 
            Q_values.append(q_val)
        
        Q_values = np.array(Q_values)
        self.Q_buffer = Q_values
        # print(self.Q_buffer)
        self.Q_index = input_data   ### in editing
        P_s = softmax(self.beta*Q_values)
        random_noise = np.random.rand(len(P_s)) * 1e-16
        P_s = P_s+random_noise
        action = np.argsort(P_s + random_noise)[::-1]

        Q_values = Q_values[action]  # 排序后的 Q_values
        self.Q_index = input_data[action]  # 排序后的 Q_index
        # print(self.Q_index)
        return action, Q_values
        # return np.argsort(P_s)[::-1]
    
    # --------- learning --------- #

    def update_V(self,r):

        # get data 
        #print(self.Q_buffer)
        coefficient = [0.5, 0, -0.5]  ## can be free parameters
        for i in range(self.Q_index.shape[0]):
            for j in range(self.Q_index.shape[1]):
                rpe = r/165 - self.Q_buffer[i]
                # rpe = r/(9*55*self.Q_index.shape[1]*self.Q_index.shape[1])
                self.W[j][self.Q_index[i][j]-1] += coefficient[int(i/3)]*self.eta*rpe
                #print(self.Q_index[i][j])
        print(self.W) 
        #exit()
        return self.W
    # def update_V(self,r, act_data):

    #     coefficient = [0.9, 0, -0.9]  ## can be free parameters
    #     for i in range(len(act_data)):
    #         for j in range(len(act_data[0])):
    #             rpe = r/9 - act_data[i]
    #             # rpe = r/(9*55*self.Q_index.shape[1]*self.Q_index.shape[1])
    #             self.W[j][act_data[i][j]-1] += coefficient[int(i/3)]*self.eta*rpe
    #             #print(self.Q_index[i][j])
    #             #print(self.W) 
    #     #exit()
    #     return self.W
    
# class fRL(baseAgent):
#     '''The feature RL'''
#     name  = 'feature RL'
#     pname = ['η', 'β']
#     pval  = [.047, 14.73]

#     # ---------- init --------- #

#     def __init__(self, nD, nF, params):
#         super().__init__(nD, nF, params)
#         self._init_W()

#     def _load_params(self, params):
#         self.eta  = params[0]
#         self.beta = params[1]

#     def _init_W(self):
#         self.W = np.ones([self.nD*self.nF])*55/9
#         self.W = self.W.reshape(self.nD,self.nF)
#         self.Q_buffer = []
#         self.Q_index = []
#     # --------- decision --------- #

#     def policy(self, input_data):

#         # Softmax(βV(s_i)): V(s_i) = ∑_fW(f)
#         Q_values = []
#         #print(input_data)
#         for i in range(input_data.shape[0]):
#             q_val = 0
#             for j in range(input_data.shape[1]):
#                 # print(input_data[i])
#                 q_val += (self.W[j][input_data[i][j]-1]) 
#             Q_values.append(q_val)
        
#         Q_values = np.array(Q_values)
#         self.Q_buffer = Q_values
#         # print(self.Q_buffer)
#         self.Q_index = input_data  
#         P_s = softmax(self.beta*Q_values)
#         # sequence = np.argsort(P_s)[::-1]
#         random_noise = np.random.rand(len(P_s)) * 1e-8
#         action = np.argsort(P_s + random_noise)[::-1]
#         return action
#         # return sequence # ,Q_values[np.argsort(P_s)[::-1]]
    
#     # --------- learning --------- #

#     def update_V(self,r):

#         # get data 
#         #print(self.Q_buffer)
#         coefficient = [0.5, 0, -0.5]  ## can be free parameters
#         print(self.Q_index)
#         for i in range(self.Q_index.shape[0]):
#             for j in range(self.Q_index.shape[1]):
#                 rpe = r/9 - self.Q_buffer[i]   # coefficient[int(i/3)]
#                 self.W[j][self.Q_index[i][j]-1] += coefficient[int(i/3)]*self.eta*rpe 
                
#                 # print(self.Q_index[i][j]-1)
#                 # print(self.Q_buffer)
#                 print(self.W) 
#         #print(self.W)
#         # exit()
#         return self.W

class bayes(fRL):
    '''The bayesian learning model'''
    name  = 'bayesian learning'
    pname = ['β']
    pval  = [4.34]

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)
        self._init_p_F()

    def _load_params(self, params):
        self.beta = params[0]

    def _init_p_F(self):
        # The probability of each feature being 
        # the target feature is initialized at 1/9 
        # at the beginning of a game 
        
        self.p_F = np.ones([self.nD*self.nF])/(self.nD*self.nF)
       
        self.p_F = self.p_F.reshape(self.nD,self.nF)

    # --------- decision --------- #

    def policy(self, input_data):
        
        Q_values = []
        #print(input_data)
        for i in range(input_data.shape[0]):
            q_val = 0
            for j in range(input_data.shape[1]):
                # print(input_data[i])
                q_val += (self.p_F[j][input_data[i][j]-1]) 
            Q_values.append(q_val)
        
        Q_values = np.array(Q_values)
        self.Q_buffer = Q_values
        # print(self.Q_buffer)
        self.Q_index = input_data   
        P_s = softmax(self.beta*Q_values)
        return np.argsort(P_s)[::-1]
    
    # --------- learning --------- #

    def update_Bel(self, r):
        
        coefficient_1 = [0.6, 0.3, 0.1]  ## can be free parameters
        if r > 0:
            coefficient_2 = [1.5, 1.1, 0.9]
        elif r < 0:
            coefficient_2 = [0.8, 0.9, 1.1]
        elif r == 0:
            coefficient_2 = [1, 1, 1]
        for i in range(self.Q_index.shape[0]):
            for j in range(self.Q_index.shape[1]):
                rpe = coefficient_2[int(i/3)]   # coefficient[int(i/3)]
                #print(rpe)
                self.p_F[j][self.Q_index[i][j]-1] = rpe*self.p_F[j][self.Q_index[i][j]-1]
                #print(self.p_F)
                #print(self.Q_index[i][j])
                #print(self.W) 
        
        self.p_F = self.p_F / np.sum(self.p_F)
        #print(self.p_F)
        #exit()
        return self.p_F

class hybrid(bayes):
    '''Hybrid Bayesian-fRL model

    This model combines the fRL with "dimnesional
    attention weights" dervied from the Bayesian model. 
    '''
    name  = 'Hybrid Bayesian-fRL'
    pname = ['η', 'α', 'β']
    pval  = [.398, 0.340, 14.09]

    def _load_params(self, params):
        self.eta   = params[0]
        self.alpha = params[1] 
        self.beta  = params[2]

    # --------- decision --------- #

    def policy(self, stims):
        
        # featurize the input sitmuli
        ss_fea = self.s_embed(stims)
       
        # the probabilities of each feature is the target are summed
        # across all features of a dimension and rased to the power
        # of α to derive dimensional attention weights
        phi_unnorm = self.p_F.reshape([self.nD, self.nF]
                ).sum(-1, keepdims=True)**self.alpha # nDx1
        self.phi = phi_unnorm / phi_unnorm.sum() # nDx1

        # V(s) = ∑d w(f_d)
        self.v_stims = np.array([(self.phi*(self.W*s_fea
                        ).reshape([self.nD, self.nF])).sum()
                        for s_fea in ss_fea]) # nDx1 x nDxnF = nDxnF
    
        # Softmax(βV(s_i))
        return softmax(self.beta*self.v_stims)
    
    # --------- learning --------- #

    def learn(self):
        self.update_Bel()
        self.update_V()

    def update_V(self):
        
        # get data 
        stims, a, r = self.mem.sample('s', 'a', 'r')
        f_chosen = self.s_embed(stims)[a]
        v_chosen = self.v_stims[a]

        # update: W(f_chosen) = W(f_chosen) + η(r-V)*Φ
        rpe = r - v_chosen
        phi_chosen = f_chosen.reshape([self.nD, self.nF])*self.phi
        self.W += self.eta*rpe*phi_chosen.reshape([-1])

class shModel(bayes):
    '''Serial Hypothesis Model

    Participants selectively attend to one feature
    at a time and over the course of several trials
    '''

    # ---------- init --------- #

    def __init__(self, nD, nF, params):
        super().__init__(nD, nF, params)
        self._init_hypo()
        self._init_p_R1F()

    def _load_params(self, params):
        self.eps   = params[0]
        self.lmbd  = params[1]
        self.theta = params[2]

    def _init_hypo(self):
        '''Randomly pick one before the task 
        '''
        f_hypo = np.random.choice(self.nD*self.nF)
        self.f_hypo = np.eye(self.nD*self.nF)[f_hypo]

    def _init_num_R1F(self):
        self.num_R1F = np.zeros([self.nD*self.nF]) 
    
    # -------- embeddings --------- #

    def s_embed(self, s):
        return [to_phi(i) for i in s]
    
    # --------- decision --------- #

    def policy(self, stims):
        '''ε-greedy policy

        until one discard this hypothesis, he chosses the
        stimulus containing the candidate feature with
        probability 1-ε and choose randomly otherwise

        '''
        n = len(stims)

        # decide if the feature is included in the stimuli
        # return 1 if included
        v_stims = np.array([(s_fea*self.f_hypo).sum()
                for s_fea in self.s_embed(stims)])
        
        return v_stims*(1-self.eps*(n+1)/n) + np.ones([n])/n

    # --------- learning --------- #

    def learn(self):
        self.update_Bel()
        self.update_hypo()

    def update_Bel(self):
        
        # Retrieve memory 
        r = self.mem.sample('r')

        self.num_R1F[self.f_hypo] += r

        # p(F) ∝ p(R|F)p(F)
        p_R1F = self.num_R1F

    def update_hypo(self):
        pass 

# ----------------------------------- #
#             Simulation              #
# ----------------------------------- #

def sim(data, 
        agent_names = ['naiveRL', 'fRL', 'fRL_decay', 'bayes', 'hybrid'], 
        seed=2023):
    '''Simulation
        simulate the models' behaviors using the mean
        parameter of the first 500 trials publised in the paper
        Niv, 2015, Table 1  
    '''
    sub_lst = data['sub_id'].unique() 
    for agent_name in agent_names:
        print(f'\nSimulating {agent_name}......')
        agent = eval(agent_name)
        sim_data = []
        for sub_id in tqdm(sub_lst):
            sub_data = data.query(f'sub_id=={sub_id}').reset_index()
            game_lst = sub_data['game'].unique()
            for game_id in game_lst:
                block_data = sub_data.query(f'game=={game_id}').reset_index()
                sim_datum = sim_block(agent, block_data, agent.pval, seed+int(sub_id))
                sim_data.append(sim_datum)
        sim_data = pd.concat(sim_data, axis=0, ignore_index=True)

        # save
        sim_data.to_csv(f'{pth}/sim_data/{agent_name}.csv', index=False)

def sim_block(agent, data, params, seed=2023):

    # init the random generator, model
    rng = np.random.RandomState(seed)
    nD, nF = 3, 3
    model = agent(nD, nF, params)

    # init simulated data 
    cols = ['choice', 'act', 'rew']
    data = data.drop(columns=cols)
    cols += ['pi']
    init_mat = np.zeros([data.shape[0], len(cols)]) + np.nan
    pred_data = pd.DataFrame(init_mat, columns=cols)

    ## loop over each row 
    for t, row in data.iterrows():

        # observe the stimuli 
        stims = [row[f's{i}'] for i in range(3)]
        rDim  = row['relevantDim']
        cPhi  = row['correctPhi']

        # make an action: a ~ π(a|s)
        pi = model.policy(stims)
        act = rng.choice(3, p=pi)

        # get reward
        choice = stims[act]
        rew = 1.*(rng.rand() < .25+.5*(str(choice)[int(rDim)-1]==cPhi))

        # record the simulated data 
        pred_data.loc[t, 'choice'] = choice 
        pred_data.loc[t, 'act']    = act
        pred_data.loc[t, 'rew']    = rew
        pred_data.loc[t, 'pi']     = pi[act]

        # memorize and learn
        mem = {'s': stims, 'a': act, 'r': rew}
        model.mem.push(mem)
        model.learn()

    return pd.concat([data, pred_data], axis=1)

# ----------------------------------- #
#             Evaluation              #
# ----------------------------------- #

def evaluate(data, 
        agent_names = ['naiveRL', 'fRL', 'fRL_decay', 'bayes', 'hybrid'], 
        seed=2023):
    '''Evaluation
        calculate the models' likelihood using the mean
        parameter of the first 500 trials publised in the paper
        Niv, 2015, Table 1  
    '''
    sub_lst = data['sub_id'].unique() 
    for agent_name in agent_names:
        print(f'\nEvaluating {agent_name}......')
        agent = eval(agent_name)
        sim_data = []
        for sub_id in tqdm(sub_lst):
            sub_data = data.query(f'sub_id=={sub_id}').reset_index()
            game_lst = sub_data['game'].unique()
            for game_id in game_lst:
                block_data = sub_data.query(f'game=={game_id}').reset_index()
                sim_datum = eval_block(agent, block_data, agent.pval)
                sim_data.append(sim_datum)
        sim_data = pd.concat(sim_data, axis=0, ignore_index=True)

        # save
        sim_data.to_csv(f'{pth}/eval_data/{agent_name}.csv', index=False)

def eval_block(agent, data, params):

    # init the random generator, model
    nD, nF = 3, 3
    model = agent(nD, nF, params)

    # init simulated data 
    cols = ['like']
    init_mat = np.zeros([data.shape[0], len(cols)]) + np.nan
    pred_data = pd.DataFrame(init_mat, columns=cols)

    ## loop over each row 
    for t, row in data.iterrows():

        # observe the stimuli 
        stims = [int(row[f's{i}']) for i in range(3)]
        act   = int(row['act'])
        rew   = row['rew']

        # calculate likelihood: π(a|s)
        pi = model.policy(stims)
        pred_data.loc[t, 'like'] = pi[act]

        # memorize and learn
        mem = {'s': stims, 'a': act, 'r': rew}
        model.mem.push(mem)
        model.learn()

    return pd.concat([data, pred_data], axis=1)

# ----------------------------------- #
#           Visualization             #
# ----------------------------------- #

def show_lr(agent_names = ['naiveRL', 'fRL', 'fRL_decay', 'bayes', 'hybrid']):

    human_data = pd.read_csv(f'{pth}/data/raw_data.csv').query('trial < 25')

    fig, axs = plt.subplots(2, 3, figsize=(9, 6), 
                sharex=True, sharey=True)
    for i, agent_name in enumerate(agent_names):
        ax = axs[i//3, i%3]
        sim_data = pd.read_csv(f'{pth}/sim_data/{agent_name}.csv'
                               ).query('trial < 25')
        sns.lineplot(x='trial', y='rew', data=human_data,
                        color=viz.Gray, ax=ax)
        sns.lineplot(x='trial', y='pi', data=sim_data,
                        color=viz.Palette[i+1], ax=ax)
        ax.plot(list(range(25)), [.33]*25, color='k', lw=1, ls='--')
        ax.set_title(agent_name)
        ax.set_ylim([.25, .7])
        ax.set_ylabel('Acc.') if i%3==0 else ax.set_ylabel('')
        ax.set_xlabel('Trial') if i//3==1 else ax.set_xlabel('')


    ax = axs[1, 2]
    ax.set_axis_off()
    fig.tight_layout()

    plt.savefig(f'{pth}/figures/learning_curves.png', dpi=300)

def show_likelihood(agent_names = ['naiveRL', 'fRL', 'fRL_decay', 'bayes', 'hybrid']):

    # collect likelihoods
    like = []
    for i, agent_name in enumerate(agent_names):
        eval_data = pd.read_csv(f'{pth}/eval_data/{agent_name}.csv'
                               ).query('trial < 25')
        like.append(eval_data['like'].mean())
    
    likelihoods = pd.DataFrame.from_dict({
        'like': like,
        'agents': agent_names
    })

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.barplot(y='agents', x='like', data=likelihoods,
                    palette=viz.Palette[1:], 
                    ax=ax)
    ax.vlines(x=.333, ymin=-.35, ymax=4.45, 
              color='k', lw=1, ls='--')
    ax.set_xlabel('Likelihood per trial')
    ax.set_ylabel('')
    fig.tight_layout()
    plt.savefig(f'{pth}/figures/likelihood.png', dpi=300)



# ------------------------------#
#          Axuilliary           #
# ------------------------------#
eps_ = 1e-13
max_ = 1e+13

def mask_fn(nA, a_ava):
    return (np.eye(nA)[a_ava, :]).sum(0, keepdims=True)

def MI(p_X, p_Y1X, p_Y):
    return (p_X*p_Y1X*(np.log(p_Y1X+eps_)-np.log(p_Y.T+eps_))).sum()

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    return np.exp(x) 

def step(w, lr):
    w.data -= lr*w.grad.data
    w.grad.data.zero_()
    
class ecpg_base_agent:
    '''Base Agent'''
    name     = 'base'
    p_bnds   = None
    p_pbnds  = []
    p_names  = []  
    p_priors = []
    p_trans  = []
    p_links  = []
    n_params = 0 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    insights = ['pol']
    
    def __init__(self, nS, nA, params):
        self.nS  = nS
        self.nA  = nA 
        self.load_params(params)
        self._init_embed()
        self._init_buffer()
        self._init_agent()
        
    def load_params(self, params): 
        return NotImplementedError
    
    def _init_embed(self):

        return NotImplementedError

    def _init_buffer(self):
        self.mem = simpleBuffer()
    
    def _init_agent(self):
        return NotImplementedError
    
    def learn(self): 
        return NotImplementedError

    def policy(self, s, **kwargs): 
        return NotImplementedError

    # --------- Insights -------- #

    def get_pol(self):
        acts = [[0, 1], [2, 3]]
        pi = np.zeros([self.nS, self.nA])
        for s in range(self.nS):
            for act in acts:
                a1, a2 = act
                pi[s, :] += self.policy(self.s2f(s), a_ava=[a1, a2])   
        return pi
    
class ECPG(ecpg_base_agent):
    '''Efficient coding policy gradient (analytical)

    We create two mathematical equivalent versions for ECPG, each
    aim at tackling different numerical problems. 
    The only difference exist in calculating sTheta.

    The analytical version, we do:
        sTheta = (u*p_a1Z.T - self.lmbda*log_dif)
    and in the fitting version, we do:
        sTheta = (u*p_a1Z.T/(self.lmdba+eps_) - log_dif)

    The analytical version is more consistent with the objective function.
    It is easy to illustrate the model behaviors with varying
    λ, because it will not divided by 0. 

    However, this implementation causes high correlation
    between parameters α_ψ and λ，bring difficulties in parameter
    estimation. 

    The same logics applied to fECPG.
    '''
    name     = 'ECPG'
    p_names  = ['alpha_psi', 'alpha_rho', 'lmbda']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 3), (-2, 3), (-6, 1.5)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_names)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_names)
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['enc', 'dec', 'pol']
    # color    = viz.Red
    marker   = '^'
    size     = 125
    alpha    = 1

    def load_params(self, params):
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.lmbda      = params[2]
        self.b          = 60

    def _init_agent(self):
        self.nZ = self.nA
        # Initialization of theta, may have some problems
        # theta = np.zeros([self.nS, self.nZ])  
        theta = np.eye(self.nS)
        # theta = np.random.rand(self.nS,self.nZ)
        self.theta = deepcopy(theta)
        self.phi   = np.zeros([self.nZ, self.nA]) 
        self._learn_pZ()

    def _learn_pZ(self):
        self.F = np.eye(self.nA)  ###
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        self.p_S   = np.ones([self.nS, 1]) / self.nS  # nSx1 
        self.p_Z   = self.p_Z1S.T @ self.p_S


    def policy(self, f, index, **kwargs):
        p_Z1s = softmax(f@self.theta)
        p_A1Z = softmax(self.phi, axis=1)
        # renormalize to avoid numeric problem
        pi = p_Z1s@p_A1Z
        # for i, _ in enumerate(pi):
        #     if f[i] !=1:
        #         pi[i] = 0
        indices = np.array(index) - 1
        new_pi = np.zeros_like(pi)
        new_pi[indices] = pi[indices]
        sequence_index = (np.argsort(new_pi)[::-1]+1)[:9]
        a_index_map = {val: idx for idx, val in enumerate(index)}
        final_sequence = [a_index_map[val] for val in sequence_index]

        return final_sequence, self.theta
        
    def learn(self, f, reward):
        self._learn_enc_dec(f, reward)
        self._learn_pZ()

    def _learn_enc_dec(self, f, reward):
    
        p_Z1s = softmax(f@self.theta)
        p_A1Z = softmax(self.phi, axis=1)
        p_a1Z = softmax(p_Z1s@p_A1Z).reshape([-1])
        u = np.array([reward - self.b])[:, np.newaxis] 

        # note: in derviation we wrote 
        #          log_dif = log p(Z|S) - log p(Z) - 1
        # However, substracting the constant 1 will not affect the 
        # numerical value of gradeint, due to the normalization 
        # term in calculating gTheta.
        log_dif = np.log(p_Z1s+eps_)-np.log(self.p_Z.T+eps_) 
        # print(log_dif.shape) 
        sTheta = (u*p_a1Z.T - self.lmbda*log_dif)
        # gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*sTheta - p_Z1s@sTheta.T))
        gTheta = -sTheta*p_Z1s*(1-p_Z1s)
        
        sPhi = u*p_Z1s.T
        # gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi
        gPhi = -p_a1Z*(np.eye(self.nA) - p_a1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi
    
    # --------- some predictions ----------- #
        
    def get_i_SZ(self):
        psi_Z1S = softmax(self.F@self.theta, axis=1)
        return MI(self.p_S, psi_Z1S, self.p_Z)
    
    def get_i_ZA(self):
        rho_A1Z = softmax(self.phi, axis=1)
        p_A = (self.p_Z.T @ rho_A1Z).T
        return MI(self.p_Z, rho_A1Z, p_A)
        
    def get_enc(self):
        return softmax(self.F@self.theta, axis=1)
    
    def get_dec(self):
        acts = [[0, 1], [2, 3]]
        rho  = 0
        for act in acts:
            m_A   = mask_fn(self.nA, act)
            p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
            p_A1Z  /= p_A1Z.sum(1, keepdims=True)
            rho += p_A1Z
        return rho 
    
class fECPG(ECPG):

    def __init__(self, nf, nS, nA, F, params):
        self.nf = nf
        self.nS  = nS
        self.nA  = nA 
        self.F = F
        self.load_params(params)
        self._init_embed()
        self._init_buffer()
        self._init_agent()

    def _init_agent(self):
        self.nZ = self.nA
        # Initialization of theta, may have some problems
        # theta = np.zeros([self.nf, self.nZ])  
        theta = np.random.rand(self.nf, self.nZ)  
        # theta = np.eye(self.nS)
        self.theta = deepcopy(theta)
        self.phi   = np.zeros([self.nZ, self.nA]) 
        self._learn_pZ()

    def _learn_pZ(self):
        # self.F = np.eye(self.nS)  ###
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        # print(self.F.shape)
        # print(self.p_Z1S.shape)
        self.p_S   = np.ones([self.nS, 1]) / self.nS  # nSx1 
        # print(self.p_S.shape)
        self.p_Z   = self.p_Z1S.T @ self.p_S
        # print(self.p_Z.shape)


if __name__ == '__main__':

    # load data
    data = pd.read_csv(f'{pth}/data/raw_data.csv', index_col=0)

    # generate simulate data
    sim(data)
    show_lr()

    # evaluate the model
    evaluate(data)
    show_likelihood()
