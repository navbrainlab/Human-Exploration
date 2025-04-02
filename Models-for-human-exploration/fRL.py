import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from tqdm import tqdm

from scipy.special import softmax
import argparse
from env.game_env import Env
import agents
import ast
#from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Example flag usage")
parser.add_argument("--env_path", type=str, default='./env/')
parser.add_argument("--game_dim", type=int, default=3)
parser.add_argument("--phase", type=str, default='P1')
parser.add_argument("--input_form", type=str, default=None)
parser.add_argument("--pretrained_steps", type=int, default=0)
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

input_form = args.input_form
input_form = '333'
pretrained_steps = args.pretrained_steps
pretrained_steps = 6
print(pth)
data = env.load_round_env(phase, pth)
dim, steps = data.shape

pval  = [0.01, 10]
#np.random.seed(1)
policy = agents.fRL(nD, nF, pval)

all_unique_values = pd.unique(data.values.ravel())
num_q = all_unique_values.shape[0]
Q_table = np.zeros(num_q)

total_step = 0
reward_list = []
while(True):
    for i in range(steps):
        input_data = data[f'{i}'].values
        
        #print(input_data)
        #exit()
        #print(input_data)
        indices = np.where(np.isin(all_unique_values, input_data))
        converted = np.array([ast.literal_eval(item) for item in input_data])
        np.random.shuffle(converted)

    
        if total_step<pretrained_steps:
            if input_form == '333':
                head1 = (total_step)%nF + 1
                head2 = (total_step+1)%nF +1
                head3 = (total_step+2)%nF +1
                #print(head1, head2, head3)
                act_data = converted[converted[:, int(total_step/nF)]==head1]
                #print(act_data)
                act_data = np.append(act_data, converted[converted[:, int(total_step/nF)]==head2], axis = 0)
                act_data = np.append(act_data, converted[converted[:, int(total_step/nF)]==head3], axis = 0)
                action= policy.policy(act_data)
            elif input_form == '322':
                head1 = (total_step)%nF + 1
                head2 = (total_step+1)%nF +1
                head3 = (total_step+2)%nF +1
                act_data = converted[converted[:, int(total_step/nF)]==head1]
                act_data = np.append(act_data, converted[converted[:, int(total_step/nF)]==head2], axis = 0)
                act_data = np.append(act_data, converted[converted[:, int(total_step/nF)]==head3], axis = 0)
                index = [0, 1, 2, 3, 4, 6, 5, 7, 8]
                act_data = act_data[index]
                action= policy.policy(act_data)
                #print(act_data)
        else:
            action= policy.policy(converted)
            act_data = converted[action]
            #exit()

        if total_step ==0:
            print('All actions:')
        elif total_step > 30:
            print(f'reward record: \n{reward_list}')
            exit()
        print(act_data.tolist())
        act_index = indices[0][action]
        reward = env.calc_reward(act_data)
        reward_list.append(reward)
        W_val = policy.update_V(reward)
        #print(f'W_val is {W_val}')
        #print(act_index)
        total_step +=1
        if reward == 100:
            print(f'total steps: {total_step}')
            #print(Q_table)
            print(f'W_val: \n{W_val}')
            print(f'final action: \n{act_data}')
            print(f'reward record: \n{reward_list}')
            #print(all_unique_values)
            exit()
