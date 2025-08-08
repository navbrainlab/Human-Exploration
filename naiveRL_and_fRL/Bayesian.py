import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from tqdm import tqdm

from scipy.special import softmax
import argparse
from env.game_env import Env
import agents_old_version
import ast
#from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Example flag usage")
parser.add_argument("--env_path", type=str, default='./env/')
parser.add_argument("--game_dim", type=int, default=3)
parser.add_argument("--phase", type=str, default='P1')
args = parser.parse_args()

env = Env()
phase = args.phase
phase = 'P2'
#embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')




if phase == 'P1':
    pth = args.env_path + '3_Dimension.csv'
    nD, nF = 3, 3
if phase == 'P2':
    pth = args.env_path + '4_Dimension.csv'
    nD, nF = 4, 3

print(pth)
data = env.load_round_env(phase, pth)
dim, steps = data.shape

pval  = [10]
np.random.seed(1)

policy = agents_old_version.bayes(nD, nF, pval)

all_unique_values = pd.unique(data.values.ravel())
num_q = all_unique_values.shape[0]
Q_table = np.zeros(num_q)

total_step = 0
reward_all = []
while(True):
    for i in range(steps):
        input_data = data[f'{i}'].values
        np.random.shuffle(input_data)
        #print(input_data)
        indices = np.where(np.isin(all_unique_values, input_data))
        converted = np.array([ast.literal_eval(item) for item in input_data])
        #Q_table_small = Q_table[indices]
        #print(converted)
        action = policy.policy(converted)
        #print(action)
        #action = np.argsort(action)
        act_data = converted[action]
        act_index = indices[0][action]
        #print(act_index)
        reward = env.calc_reward(act_data)
        reward_all.append(reward)
        reward_mean = np.mean(reward)
        real_reward = reward - 75
        print(reward)
        
        #print(Q_table)
        p_F = policy.update_Bel(real_reward)
        print(f'p_F is {p_F}')
        #print(act_index)
        total_step +=1
        if reward == 100 or total_step == 50:
            print(f'total steps is {total_step}')
            #print(Q_table)
            print(all_unique_values)
            exit()



