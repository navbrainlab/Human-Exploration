import numpy as np
import pandas as pd
import os
import scipy.stats
from itertools import combinations
from taskSetting_jilab import hypotheses, rewardSetting
import random
import ast
import pic_draw

# 本轮给agent的九个套餐
def get_options(trial, phase):
    if phase == "p1":
        options = options_3D.iloc[:, trial].tolist()
    elif phase == "p2":
        options = options_4D.iloc[:, trial].tolist()
    return options
# options = get_options(1,'p1')
# print(options)



def score_func(option,h,phase):
    if phase == 'p1':
        return np.nansum([h[i][int(option[i]) - 1] for i in range(3)])
    else:
        return np.nansum([h[i][int(option[i]) - 1] for i in range(4)])

#将h转换成当前options下的action
def hypothesisToAction(h,options):
    # h = [[1,nan,nan],[nan,nan,nan], [nan,nan,nan]]
    random.shuffle(options)
    options_score_sort = sorted(options, key=lambda opt: score_func(opt, h, phase), reverse=True)
    # 将排序后的列表分成 3 组，每组 3 个列表
    options_score_sort = [options_score_sort[i:i + 3] for i in range(0, len(options_score_sort), 3)]
    return options_score_sort


def getColumnIdx(h, hypothesized_action, action, row_idx):
    # h = [[1,nan,nan],[nan,nan,nan], [nan,nan,nan]]
    matchedCount = 0 # the column idx we want

    # 把没有hypothesized relevant feature的options变成[0,0,0]，在和action比较的时候略去，action和hypothesis比较的时候只看有relevant的
    def find_indices_with_1(h): # 找到所有relevant值的(dim, feature)索引
        indices = []
        for i, row in enumerate(h):  
            for j, value in enumerate(row):
                if value == 1:
                    indices.append((i, j)) 
        return indices
    
    def count_matching_elements(hypothesized_action, indices):
        filteredAction = {}  # 存储每个子列表中满足条件的组合和对应的index
        # 遍历 hypothesized_action 中的每个大子列表
        for group in hypothesized_action:
            for ha in group:  # 遍历当前组中的每个列表
                count = 0
                for i, j in indices:  # 遍历所有 (i, j) 索引对
                    if ha[i] == j + 1:  # 判断第 i 个元素是否等于 j + 1
                        count += 1  # 如果满足条件，计数加 1
                filteredAction[tuple(ha)] = count
        return filteredAction
    filteredAction = count_matching_elements(hypothesized_action, indices = find_indices_with_1(h))
    # print(filteredAction,'\n')
    print('this is the trial number', trial)
    print('this is the row index', row_idx)
    if row_idx==0: # for 1D relevant games
        for ha, count in filteredAction.items():
            ha = list(ha)
            if count==1:
                if ha in action[0]:
                    matchedCount += 1
    elif row_idx==1:
        iterated_count = 0
        row_1_count = 0
        for ha, count in filteredAction.items():
            ha = list(ha)
            if count==2 and list(ha) in action[0]:
                matchedCount += 1
            elif count==1:
                # row_1_count += 1
                if list(ha) in action[0]:
                   if row_1_count<=2:
                       row_1_count += 1
                       matchedCount += 1
                elif list(ha) in action[1]:
                    iterated_count += 1
                    if iterated_count<=2:
                        matchedCount += 1

    elif row_idx==2:
        has_count_3 = any(count == 3 for ha, count in filteredAction.items())
        if has_count_3:
            row_3_count = 0
            row_1_count = 0
            for ha, count in filteredAction.items():
                ha = list(ha)
                if count==3:
                   if list(ha) in action[0]:
                        matchedCount += 1
                elif count==1:
                    if list(ha) in action[0]:
                        if row_1_count<2:
                            row_1_count += 1
                            matchedCount += 1
                    elif list(ha) in action[1]:
                        matchedCount += 1
                    # elif list(ha) in action[2]:
                    #     if row_3_count==0:
                    #         row_3_count += 1
                    #         matchedCount += 1

        else:
            for ha, count in filteredAction.items():
                ha = list(ha)
                if count==2 and list(ha) in action[0]:
                    matchedCount += 1
                elif count==1 and list(ha) in action[1]:
                    matchedCount += 1

        # for ha, count in filteredAction.items():
        #     ha = list(ha)
        #     if count>1 and list(ha) in action[0]:
        #         matchedCount += 1
        #     elif count==1 and list(ha) in action[1]:
        #         matchedCount += 1
    print('matchedCount:', matchedCount)
    return matchedCount



# get_action: options -> calculate Expected Reward -> pick an action
# options -> allchoices
def generate_choices(options):
    allChoices = []
    seen = set()

    for group1 in combinations(options, 3):
        remaining1 = [x for x in options if x not in group1]
        for group2 in combinations(remaining1, 3):
            group3 = [x for x in remaining1 if x not in group2]

            # 用 tuple(tuple(...)) 做 hashable 的去重 key
            g1 = [list(x) for x in group1]
            g2 = [list(x) for x in group2]
            g3 = [list(x) for x in group3]

            key = (tuple(map(tuple, g1)), tuple(map(tuple, g2)), tuple(map(tuple, g3)))
            if key not in seen:
                seen.add(key)
                allChoices.append([g1, g2, g3])

    return allChoices

# calculate pHypothesis
# def trialLikelihood(hypotheses, action, reward, rewardSetting = rewardSetting):
#     lik = np.zeros(len(hypotheses))
#     for iH, h in enumerate(hypotheses):
#         hypothesized_action = hypothesisToAction(h, options)
#         row_idx = np.sum(~np.isnan(h))-1  # rewardSetting里第几行 -> #dim relevant
#         # column_idx = np.sum(np.equal(hypothesized_action, action)) 
#         column_idx = getColumnIdx(h, hypothesized_action, action, row_idx) # rewardSetting里第几列 -> #similar elements
#         pReward = rewardSetting[row_idx][column_idx]
#         # print(row_idx, column_idx, pReward,'\n', h, '\n', hypothesized_action, '\n', action,'\n---------------------------------------------------\n')
#         rewardWeight = reward/100
#         lik[iH] = pReward if reward>70 else (1-pReward) # bayesian_data_3, falsely exclude the reward weight
#         # lik[iH] = (pReward*rewardWeight) if reward>70 else pReward*0.2 # maybe bayesian_data_2
#         lik[iH] = pReward * (reward/100) # bayesian_data_1, only this line
#     return lik,column_idx
def trialLikelihood(hypotheses, action, reward, last_p, rewardSetting = rewardSetting):
    lik = np.zeros(len(hypotheses))
    similarity = 0
    max_lik = 0
    all_column_idx = np.zeros(len(hypotheses))
    all_pReward = np.zeros(len(hypotheses))
    all_hypothesis_action = []
    for iH, h in enumerate(hypotheses):
        hypothesized_action = hypothesisToAction(h, options)
        all_hypothesis_action.append(hypothesized_action)

    pRewardMat = np.zeros((len(hypotheses),len(all_hypothesis_action))) 

    for iH, h in enumerate(hypotheses):
        row_idx = np.sum(~np.isnan(h))-1  # rewardSetting里第几行 -> #dim relevant
        # column_idx = np.sum(np.equal(hypothesized_action, action)) 
        hypo_id = hypotheses.index(h)  # 获取当前hypothesis的索引
        print('here comes a new h', hypo_id)
        column_idx = getColumnIdx(h, hypothesized_action, action, row_idx) # rewardSetting里第几列 -> #similar elements
        pReward = rewardSetting[row_idx][column_idx]
        # print(row_idx, column_idx, pReward,'\n', h, '\n', hypothesized_action, '\n', action,'\n---------------------------------------------------\n')
        # rewardWeight = reward/100
        # lik[iH] = pReward if reward>70 else (1-pReward) # bayesian_data_3, falsely exclude the reward weight
        # lik[iH] = (pReward*rewardWeight) if reward>70 else pReward*0.4 # maybe bayesian_data_2
        lik[iH] = pReward if reward>70 else 1-pReward
        # lik[iH] = (pReward*rewardWeight) if reward>70 else pReward*0.2 # maybe bayesian_data_2
        # lik[iH] = pReward * (reward/100) # bayesian_data_1, only this line
        all_column_idx[iH] = column_idx
        all_pReward[iH] = pReward

        # compare and calculate a column_idx for each hypothesis-action pair
        for iChoice, _ in enumerate(all_hypothesis_action):
            column_idx2 = getColumnIdx(h, hypothesized_action, all_hypothesis_action[iChoice], row_idx)
            pRewardMat[iH,iChoice] = rewardSetting[row_idx][column_idx2]
    
    print(pRewardMat)
    # print(pRewardMat.shape)


    
    p = lik * last_p
    # pHypothesis = p / np.sum(p)
    p = p / np.sum(p)  # 归一化为概率分布
    
    return lik,all_column_idx,all_hypothesis_action,all_pReward, p, pRewardMat




# calculate pReward
# def pRewardMatrix(hypotheses, action, allChoices, rewardSetting):
#     pRewardMat = np.zeros((len(hypotheses),len(allChoices)))
#     for iH, h in enumerate(hypotheses):
#         hypothesized_action = hypothesisToAction(h, options)
#         for iChoice, choice in enumerate(allChoices):
#         #     # stimuli, _ = choiceToStimuli(choice)
#             row_idx = np.sum(~np.isnan(h))-1
#             column_idx = getColumnIdx(h, hypothesized_action, choice, row_idx)
#             pRewardMat[iH,iChoice] = np.mean([rewardSetting[row_idx][column_idx] for choice in allChoices])
#     return pRewardMat

# calculate Expected reward
def ER_calculation(pHypothesis, pReward, beta):
    p = pHypothesis
    ER = np.mean(pReward.T * p, axis=0)

    return ER

# 这是在计算这个choice的期望reward
def rewardFunc(action): # modified on june 17
    # action:本轮的选择，九个套餐在三行中的排列
    # 数据结构:   [    [[1,2,3],[, , ],[, ,]],
    #                 [[, ,],[, ,],[, , ]],
    #                 [[, ,],[, ,],[, , ]]  ]
    '''calculate the reward'''
    level_order_reward = {'1': 10, '2': 5, '3': 1}
    block_reward = []
    reward_dim =[0,1]


    # Calculate rewards for each block action
    Main_dim_1_reward = 0
    Main_dim_2_reward = 0
    for i in action: #[[1,2,3],[, , ],[, ,]]
        for j in i: #[1,2,3]
            Main_dim_1_reward = 0
            Main_dim_2_reward = 0
            Main_dim_1_reward += level_order_reward[str(j[reward_dim[0]])]
            Main_dim_2_reward += level_order_reward[str(j[reward_dim[1]])]
            block_reward.append(Main_dim_1_reward + Main_dim_2_reward)
    block_reward = np.array(block_reward)
    block_reward = block_reward.reshape(3, 3)

    # Apply weights to block rewards
    weighted_block_reward = (np.array(block_reward).T * np.array([10, 5, 1])).T

    # Calculate final reward
    all_reward = (sum(sum(weighted_block_reward)) - 350) * 90 / 324 + 10

    # Add noise to the reward if it's not 0 or 100
    if all_reward not in [0, 100]:
        noise = np.random.uniform(-2, 2)
        all_reward += noise

    reward = round(all_reward)  # Final reward
    return reward




# 怎么把九个套餐排列获得action（choice -> action主函数）
# 根据上一轮经验得到的allChoices选择概率，确定当前轮的action
def get_action(last_p, last_pRewardMat, allHypothesizedActions):

    # lik,similarity,all_hypothesis_action,all_pReward, pRewardMat = trialLikelihood(hypotheses, action, reward=rewardFunc(action))

    # print('\nthis is p hypotheses\n',pHypothesis)
    # pReward = pRewardMatrix(hypotheses, action, allChoices, rewardSetting) #对每一个h都遍历了所有的choice，得到 h x choice 维向量 
    # print('this is pr------------------\n', pReward)
    beta = 1.0
    ER = ER_calculation(last_p, last_pRewardMat, beta)
    # print('\nthis is ER !!!!!!!!!!!!!!!!!!!!!!!!!!!\n',ER)
    # print(ER.shape)
    ER = ER / np.sum(ER)  # 归一化为概率分布
    ER = ER.flatten()
    # action = allChoices[np.random.choice(np.arange(ER), size=1, p=ER)[0]]
    # idx = np.argmax(ER)

    # 找到所有最大值对应的索引
    max_value = np.max(ER)
    max_indices = np.where(ER == max_value)[0]  # 返回所有最大值的索引

    # 从最大值索引中随机选择一个
    idx = np.random.choice(max_indices)

    h_idx = np.argmax(last_p)
    # idx = np.random.choice(len(allChoices), size=1, p=ER)[0]
    # action = allChoices[idx]
    this_trial_action = allHypothesizedActions[idx]


    # for iH, h in enumerate(hypotheses):
    #     hypothesized_action = hypothesisToAction(h, options)
    #     row_idx = np.sum(~np.isnan(h))-1  # rewardSetting里第几行 -> #dim relevant
    #     # column_idx = np.sum(np.equal(hypothesized_action, action)) 
    #     column_idx = getColumnIdx(h, hypothesized_action, action, row_idx) 

    # return lik, pHypothesis, this_trial_action, idx, h_idx, similarity, ER,all_hypothesis_action,all_pReward
    return this_trial_action, idx, h_idx, ER

def define_choice_pattern(phase, agent_dict, act_data):  # 把所有维度这一轮的选择模式加入列表

    for i in range(nD):
        dim_list = 'pattern_' + str(i + 1)
        choice_seq_list = 'choice_' + str(i + 1)
        # print(type(act_data))
        act_data = np.array(act_data)
        
        dim_choice = act_data[:,:, i]
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





if __name__=='__main__':
    for agent in range(100):
        # prev_p = {i: [] for i in range(0, 63)}
        reward = 0

        last_p = np.full(len(hypotheses), 1 / len(hypotheses)) # 初始化p的值



        phase = 'p1'
        nF = 3
        if phase == 'p1':
            nD = 3
        else:
            nD = 4

        save = 'nsy'
        # save = 'wyl'
        if save == 'wyl':
            save_pth = os.path.dirname('C:/Users/Windows11/Desktop/humans-combine-value-learning-and-hypothesis-testing-main/humans-combine-value-learning-and-hypothesis-testing-main/NEWdata/fitting.csv')  # 提取文件夹路径
        else:
            save_pth = 'C:/Users/Windows11/Desktop/bayesian_data'
            save_pth = save_pth + '/'+str(agent)+'_fitting.csv'
            folder_path = os.path.dirname(save_pth)
        if not os.path.exists(folder_path):           # 检查文件夹是否存在
            os.makedirs(folder_path) 
            
        # print(save_pth)
        data_columns = ['trial', 'phase', 'action', 'reward', 'likelihood_vector_for_h',
                        'p_for_h','similarity','best_h_idx','action_idx','choose_h','choose_action',
                        'hypothesized_action','ER','all_hypothesis_action','all_hypothesis_action_1','all_pReward'] 
        df = pd.DataFrame(columns = data_columns) 
        df.to_csv(save_pth, index=False)

        # get options from data files
        options_3D = pd.read_csv("humans-combine-value-learning-and-hypothesis-testing-main\data\options_3_Dimension.csv")
        options_4D = pd.read_csv("humans-combine-value-learning-and-hypothesis-testing-main\data\options_4_Dimension.csv")

        score_of_all_food = {'0':[0]*3,'1':[10,5,1],'2':[10,5,1],'3':[0]*3}

        draw_dict = {'agent_id':[],'round':[],'score':[],
                'choice_1':[],'choice_2':[],'choice_3':[],
                'pattern_1':[],'pattern_2':[],'pattern_3':[],
                'pos_1':[],'pos_2':[],'pos_3':[]}
        
        all_best_hypothesis_idx = []
        p = -1
        for trial in range(30):
            print('agent:', agent, 'trial:', trial)

            if reward ==100:
                break
            else:
                draw_dict['agent_id'].append(agent)
                draw_dict['round'].append(trial)
                
                # 得到options
                options = get_options(trial, phase)
                options = [ast.literal_eval(x) for x in list(options)]
                allChoices = generate_choices(options)
                allHypothesizedActions = []
                for iH, h in enumerate(hypotheses):
                    hypothesized_action = hypothesisToAction(h, options)
                    allHypothesizedActions.append(hypothesized_action)

                # if reward == 0:
                #             # print(len(allChoices))
                #     action_idx = np.random.choice(len(allChoices))
                #     action = allChoices[action_idx]

                # 根据上一轮的action和reward，更新likelihood，更新hypothesis和概率和ER,做出action
                [last_p, last_pRewardMat] = [p, pRewardMat] if trial > 0 else [last_p, np.full((len(hypotheses), len(hypotheses)), 1 / len(hypotheses))]
                # lik, pHypothesis, action, action_idx, hypothesis_idx,similarity, ER,all_hypothesis_action,all_pReward = get_action(last_p, last_pRewardMat) #单轮的action
                action, idx, h_idx, ER = get_action(last_p, last_pRewardMat, allHypothesizedActions) #单轮的action
                # 保证tllh 里使用的action是本轮的，get_action得到的
                reward=rewardFunc(action)
                lik,similarity,all_hypothesis_action,all_pReward, p, pRewardMat = trialLikelihood(hypotheses, action, reward, last_p, rewardSetting = rewardSetting)

                all_best_hypothesis_idx.append(h_idx)
                
                # 得到对应的reward
                draw_dict = define_choice_pattern(phase,draw_dict,action)
                

                draw_dict['score'].append(reward)
                print('this is the reward:\n',reward,'\n')
                
                # 把reward和action记录下来
                row = {
                    "trial": trial+1,
                    "phase": phase,
                    "action": action,
                    "reward": reward,
                    "likelihood_vector_for_h": str(lik),
                    'p_for_h':str(p),
                    'similarity':similarity,
                    'best_h_idx':h_idx,
                    'action_idx':idx,
                    
                    'choose_h':hypotheses[h_idx],
                    'choose_action': hypotheses[idx],
                    # 'hypothesized_action':allHypothesizedActions[action_idx],
                    'hypothesized_action':action,

                    'ER': ER,
                    'all_hypothesis_action': str(all_hypothesis_action),
                    'all_hypothesis_action_1': str(allHypothesizedActions),
                    'all_pReward': str(all_pReward)

                    }
                df = pd.DataFrame([row])  # 将字典转为DataFrame
                # print(save_pth)
                df.to_csv(save_pth, mode="a", header=False, index=False)  # 追加写入文件

        # print(all_best_hypothesis_idx)   # 每一轮最佳hypothesis的index
        # print(draw_dict)
        pic_draw.choice_reward_pic('bayesian_data',3,draw_dict)
            # pic_draw.choice_reward_pic('naive_RL_0615',nD,agent_dict)
            # draw_dict = define_choice_pattern(phase,draw_dict,act_data)