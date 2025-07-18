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
import pic_draw
import copy
#from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Example flag usage")
parser.add_argument("--env_path", type=str, default='./env/')
parser.add_argument("--game_dim", type=int, default=3)
parser.add_argument("--phase", type=str, default='P2')
parser.add_argument("--input_form", type=str, default=None)
parser.add_argument("--pretrained_steps", type=int, default=0)
args = parser.parse_args()

env = Env()
phase = args.phase
phase = 'P1'
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

pval  = [0.05, 10]
#np.random.seed(1)
policy = agents.fRL(nD, nF, pval)

all_unique_values = pd.unique(data.values.ravel())

score_of_food = {'dim1': [0] * nF, 'dim2': [10, 5, 1], 'dim3': [10, 5, 1], 'dim4': [0] * nF}


# 计算每个食物组合的得分并将all_unique_values按分数排序


def score(group):
    group = ast.literal_eval(group)

    return score_of_food['dim2'][int(group[1]) - 1] +  score_of_food['dim3'][int(group[2]) - 1]


all_unique_values = sorted(all_unique_values, key=score, reverse=False)

score_list = []
for series in all_unique_values:
    series_score = score(series)
    score_list.append(series_score)
all_unique_values = np.array(all_unique_values)
score_list = np.array(score_list)
all_unique_values_df = pd.DataFrame({'combine':all_unique_values,'score':score_list})
all_unique_values_df.to_csv(r"C:\Users\Windows11\Desktop\all_unique_values_fRL.csv", index=False)

num_q = all_unique_values.shape[0]
Q_table = np.zeros(num_q)

total_step = 0
reward_table = []


def define_choice_pattern(agent_dict,act_data): #把所有维度这一轮的选择模式加入列表
    for i in range(nD):
        dim_list = 'pattern_' +str(i+1)
        choice_seq_list = 'choice_' +str(i+1)
        dim_choice = act_data[:, i]
        arr = np.array(dim_choice).reshape(-1, 3)

        # 对每一行排序
        sorted_arr = np.sort(arr, axis=1)

        # 再展平成一维 list
        dim_choice = sorted_arr.flatten().tolist()
                
        all_row_count = ''
        choice_pattern = []
        for food in range(nF):
            
            max_food_count = 0

            for j in range(3): #看每一行
                # print(j)
                row_count = len([x for x in dim_choice[j*3:j*3+3] if x == food+1])
                if row_count > max_food_count:
                    max_food_count = row_count
            choice_pattern.append(max_food_count)
            # all_row_count = all_row_count + str(max_food_count)
        all_row_count = ''.join(str(i) for i in sorted(choice_pattern))
        # choice_pattern = all_row_count
        agent_dict[dim_list].append(all_row_count)
        agent_dict[choice_seq_list].append(dim_choice)
    return agent_dict



for agent_id in range(100):
    policy = agents.fRL(nD, nF, pval)
    # Q_table = np.zeros(num_q)
    not_fullscore = True
    total_step = 0
    if phase == 'P1':
        agent_dict = {'agent_id':[],'round':[],'score':[],
                'choice_1':[],'choice_2':[],'choice_3':[],
                'pattern_1':[],'pattern_2':[],'pattern_3':[],
                'pos_1':[],'pos_2':[],'pos_3':[],'W_val':[]}
    else:
        agent_dict = {'agent_id':[],'round':[],'score':[],
                'choice_1':[],'choice_2':[],'choice_3':[],'choice_4':[],
                'pattern_1':[],'pattern_2':[],'pattern_3':[],'pattern_4':[],
                'pos_1':[],'pos_2':[],'pos_3':[],'pos_4':[],'W_val':[]}
    while not_fullscore:

# while(True):
        for i in range(steps):
            input_data = data[f'{i}'].values
            
            #print(input_data)
            #exit()
            #print(input_data)
            indices = np.where(np.isin(all_unique_values, input_data))
            converted = np.array([ast.literal_eval(item) for item in input_data])
            np.random.shuffle(converted)

        
            if total_step<pretrained_steps: #好了没事了，这个是pretrain，which不需要
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

            # if total_step ==0:
            #     print('All actions:')
            # elif total_step > 0:
            #     print(f'reward record: \n{reward_list}')
                # exit()
            # print(act_data.tolist())
            act_data = converted[action] #本轮的选择
        
            agent_dict = define_choice_pattern(agent_dict,act_data)#更新本轮的选择模式
            act_index = indices[0][action]
            reward = env.calc_reward(act_data)
            # reward_list.append(reward)
            W_val = policy.update_V(reward)
            #print(act_index)
            # print(reward)
            total_step +=1
            agent_dict['round'].append(total_step)
            agent_dict['score'].append(reward)
            agent_dict['agent_id'].append(str(agent_id))
            agent_dict['W_val'].append(copy.deepcopy(W_val))
            
            if reward == 100 or total_step==30:
                # print(f'total steps: {total_step}')
                # #print(Q_table)
                # print(f'W_val: \n{W_val}')
                # print(f'final action: \n{act_data}')
                # print(f'reward record: \n{reward_list}')
                #print(all_unique_values)
                # exit()
                pic_draw.choice_reward_pic('fRL_agent100_0624',nD,agent_dict)

                not_fullscore = False

                break
