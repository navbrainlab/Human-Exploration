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
from visualization import visualization_r_a


parser = argparse.ArgumentParser(description="Example flag usage")
parser.add_argument("--env_path", type=str, default='./env/')
parser.add_argument("--game_dim", type=int, default=3)
parser.add_argument("--phase", type=str, default='P1')
parser.add_argument("--input_form", type=str, default=None)
parser.add_argument("--pretrained_steps", type=int, default=0)
parser.add_argument("--seed", type=int, default=101)
args = parser.parse_args()

env = Env()
phase = args.phase
phase = 'P1'
seed = args.seed
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
pretrained_steps = 0
print(pth)
data = env.load_round_env(phase, pth)
dim, steps = data.shape

params  = [0.5, 0.5, 0.1]  # alpha_psi, alpha_rho, lmbda

input_data = data.values
nS = input_data.shape[1]
# print(nS)
# exit()
nS = nF**nD
nA = nS
#np.random.seed(1)
policy = agents.ECPG(nS, nA, params)
vis_func = visualization_r_a()

all_unique_values = pd.unique(data.values.ravel())
num_q = all_unique_values.shape[0]
Q_table = np.zeros(num_q)

np.random.seed(seed)
threshold_reward = False
total_step = 0
reward_list = []
action_list = []
while(True):
    for i in range(steps):
        input_data = data[f'{i}'].values
        
        #print(input_data)
        #exit()
        #print(input_data)
        indices = np.where(np.isin(all_unique_values, input_data))
        converted = np.array([ast.literal_eval(item) for item in input_data])
        np.random.shuffle(converted)
        # print(converted)

        # Covert food combination to index, similar as coverting ternary to decimalism, but need to minus ternary 110
        index = []
        if phase == 'P1':
            for ternary in converted:
                decimal = np.sum(ternary * (3 ** np.arange(len(ternary)-1, -1, -1)))-12
                index.append(decimal)
        elif phase == 'P2':
            for ternary in converted:
                decimal = np.sum(ternary * (3 ** np.arange(len(ternary)-1, -1, -1)))-39
                index.append(decimal)
        input = np.zeros(nF**nD)
        # print('index: ', index)
        for i, val in enumerate(input):
            if i+1 in index:
                input[i]+=1

    
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
                action, theta= policy.policy(act_data)
                #print(act_data)
        else:
            # print(index)
            # print(input)
            action, theta= policy.policy(input, index)
            # print(theta)
            act_data = converted[action]
            #exit()

            # Normalization
            # norm_theta = (theta - theta.min()) / (theta.max() - theta.min())
            row_sums = theta.sum(axis=1, keepdims=True)  # shape (n, 1)
            norm_theta = theta / (row_sums+1e-6)
            # Visualization
            # plt.imshow(norm_theta, cmap='viridis')
            # plt.colorbar()
            # plt.title(f"theta {total_step}")

            # plt.savefig(f'./figs/theta_{total_step}.png')  # save as matrix_0.png, matrix_1.png, ...
            # plt.close()

        if total_step ==0:
            print('All actions:')
        elif total_step >= 50:
            print(f'reward record: \n{reward_list}')
            exit()
        # print(act_data.tolist())
        act_index = indices[0][action]
        reward = env.calc_reward(act_data)

        # Option: use threshold reward
        if threshold_reward:
            if reward >= 70:
                input_reward = 1
            else:
                input_reward = 0
        else:
            input_reward = reward
        reward_list.append(reward)
        action_list.append(act_data.tolist())
        policy.learn(input, input_reward)
        
        # print(act_index)
        # print(reward)
        total_step +=1
        if total_step == 1:
            reward = 0
        if reward == 100:
            print(f'total steps: {total_step}')
            print(f'final action: \n{act_data}')
            print(f'reward record: \n{reward_list}')
            break
            #print(all_unique_values)
            # exit()
           
    # Visualization action and reward using line chart
    vis_func.update(action_list, np.array(reward_list))
    vis_func.vis()
    break       