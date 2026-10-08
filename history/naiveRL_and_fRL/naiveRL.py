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
parser.add_argument("--phase", type=str, default='P1')
args = parser.parse_args()

env = Env()
phase = args.phase
phase = 'P1'
#embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

game_dim = args.game_dim
game_dim = 3
if game_dim==3:
    pth = args.env_path + '3_Dimension.csv'
elif game_dim==4:
    pth = args.env_path + '4_Dimension.csv'

print(pth)
data = env.load_round_env(phase, pth)
dim, steps = data.shape
if phase == 'P1':
    nD, nF = 3, 3
if phase == 'P2':
    nD, nF = 4, 3

pval  = [1.0, 1.0, 0.1]
policy = agents.naiveRL(nD, nF, pval)

all_unique_values = pd.unique(data.values.ravel())
# print(type(all_unique_values))

score_of_food = {'dim1':[10,5,1],'dim2':[10,5,1],'dim3':[0]*nF,'dim4':[0]*nF}

#计算每个食物组合的得分并将all_unique_values按分数排序


def score(group):
    group = ast.literal_eval(group)

    return score_of_food['dim1'][int(group[0]) - 1] +  score_of_food['dim2'][int(group[1]) - 1]


all_unique_values = sorted(all_unique_values, key=score, reverse=False)

score_list = []
for series in all_unique_values:
    series_score = score(series)
    score_list.append(series_score)
all_unique_values = np.array(all_unique_values)
score_list = np.array(score_list)
all_unique_values_df = pd.DataFrame({'combine':all_unique_values,'score':score_list})
all_unique_values_df.to_csv(r"results/naiveRL/3D_results/all_unique_values_naiveRL.csv", index=False)



#all_unique_values是所有食物组合的array，对P2来说be like: array(['[1, 1, 1, 1]', '[1, 1, 3, 3]', '[1, 1, 1, 2]'])
num_q = all_unique_values.shape[0]
# 食物组合的总数，P2 num_q = 81

# print(all_unique_values)
#Q_table = np.zeros(num_q)
Q_table = np.ones(num_q)*50/9
#对所有食物组合的估计值的初始值

total_step = 0

# agent_dict = {'agent_id':[],'round':[],'score':[],
#               'choice_1':[],'choice_2':[],'choice_3':[],'choice_4':[],
#               'pattern_1':[],'pattern_2':[],'pattern_3':[],'pattern_4':[],
#               'pos_1':[],'pos_2':[],'pos_3':[],'pos_4':[],}
reward_table = []


# choice_reward_pair=pd.read_csv(r'C:\Users\Windows11\Desktop\choice_reward_pair.csv')
def define_choice_pattern(agent_dict,act_data): #把所有维度这一轮的选择模式加入列表
    for i in range(nD):
        dim_list = 'pattern_' +str(i+1)
        choice_seq_list = 'choice_' +str(i+1)
        # dim_choice = act_data[:, i]
        dim_choice = [row[i] for row in act_data]
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
    Q_table = np.ones(num_q) * 55 / 9
    not_fullscore = True
    all_indices = []
    total_step = 0
    # agent_dict = {'agent_id':[],'round':[],'score':[],
    #           'choice_1':[],'choice_2':[],'choice_3':[],'choice_4':[],
    #           'pattern_1':[],'pattern_2':[],'pattern_3':[],'pattern_4':[],
    #           'pos_1':[],'pos_2':[],'pos_3':[],'pos_4':[],'Q_table':[]}

    if phase == 'P1':
        agent_dict = {'agent_id': [], 'round': [], 'score': [],
                      'choice_1': [], 'choice_2': [], 'choice_3': [],
                      'pattern_1': [], 'pattern_2': [], 'pattern_3': [],
                      'pos_1': [], 'pos_2': [], 'pos_3': [], 'action':[], 'Q_table':[],'indices':[], 'spare_indices':[]}
    else:
        agent_dict = {'agent_id': [], 'round': [], 'score': [],
                      'choice_1': [], 'choice_2': [], 'choice_3': [], 'choice_4': [],
                      'pattern_1': [], 'pattern_2': [], 'pattern_3': [], 'pattern_4': [],
                      'pos_1': [], 'pos_2': [], 'pos_3': [], 'pos_4': [],'action':[], 'Q_table':[],'indices':[], 'spare_indices':[]}

    while not_fullscore:
        #对每个食物组合的Q值重新估计
    # print(steps)
        for i in range(steps):
            input_data = data[f'{i}'].values
            indices = np.where(np.isin(all_unique_values, input_data))
            all_indices += list(indices[0])
            all_indices = list(set(all_indices))
            spare_indices = [i if i not in all_indices else -1 for i in range(27)]
            # converted = np.array([ast.literal_eval(item) for item in input_data])
            Q_table_small = Q_table[indices]
            action = policy.policy(Q_table_small, i) #该动作的索引

            # act_data = converted[action] #本轮的选择
            act_index = indices[0][action]
            act_data = all_unique_values_df.loc[act_index, 'combine'].values
            act_data = [ast.literal_eval(item) for item in act_data]
        
            agent_dict = define_choice_pattern(agent_dict,act_data)#更新本轮的选择模式
        
            reward = env.calc_reward(act_data)
        
            Q_table = policy.learn(act_index, reward, Q_table)

            reward_table.append(reward)

            agent_dict['round'].append(total_step)
            agent_dict['score'].append(reward)
            agent_dict['agent_id'].append(str(agent_id))
            agent_dict['action'].append(act_data)
            agent_dict['Q_table'].append(copy.deepcopy(Q_table))
            agent_dict['indices'].append(list(indices))
            agent_dict['spare_indices'].append(spare_indices)
        
            total_step +=1
            if reward == 100 or total_step >=30 :
                pic_draw.choice_reward_pic('3D_results',nD,agent_dict)
                not_fullscore = False

                break


