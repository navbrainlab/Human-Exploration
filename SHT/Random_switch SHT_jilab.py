import numpy as np
import pandas as pd
import os
import scipy.stats
from itertools import combinations
from taskSetting_jilab import hypotheses_3D as hypotheses
# from taskSetting_jilab import hypotheses_test as hypotheses
from taskSetting_jilab import rewardSetting
import random
import ast
import pic_draw
from scipy.special import logsumexp
from copy import deepcopy

informed = False
taskCondKeys = [(False, 2)]
numDimensions = 3
gameLength = 30
keys = [(iTrial, lOld) for iTrial in range(gameLength) for lOld in range(iTrial)]
posterior = dict(zip(keys, [None for _ in range(len(keys))]))
# data = 
choiceIndex_valid = []

def define_choice_pattern(phase,agent_dict,act_data): #把所有维度这一轮的选择模式加入列表

    for i in range(nD):
        dim_list = 'pattern_' +str(i+1)
        choice_seq_list = 'choice_' +str(i+1)
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


def zerosLists(numList, lengthList):
    return [[0] * lengthList for _ in range(numList)]
choiceIndex, stimulusIndex = zerosLists(numList=2, lengthList = gameLength)



# 本轮给agent的九个套餐
def get_options(trial, phase):
    if phase == "p1":
        options = options_3D.iloc[:, trial].tolist()
    elif phase == "p2":
        options = options_4D.iloc[:, trial].tolist()
    return options

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

random_switch = True

def emptyLists(numList):
    return [[] for _ in range(numList)]
def zerosLists(numList, lengthList):
    return [[0] * lengthList for _ in range(numList)]
def emptyDicts(numDict, keys, lengthList):
    if lengthList == 0:
        dicts = [dict(zip(keys, emptyLists(numList=len(keys)))) for _ in range(numDict)]
    else:
        dicts = [dict(zip(keys, zerosLists(numList=len(keys), lengthList=lengthList))) for _ in range(numDict)]

    if numDict > 1:
        return dicts
    elif numDict == 1:
        return dicts[0]

def calculate_24terms(t, logPhSwitchmodelLast=None, logPhOldLast=None, logPlOldLast=None, logPlRunlengthmodelLast=None, logPchLast=None):
    if t == 1:

        # logPhOld = logPchLast + logPhOldLast
        logPhOld = logPhOldLast
        norm = logsumexp(logPhOld)
        if not np.isinf(norm):
            logPhOld = logPhOld - norm
        logPhOld = logPhOld[:, np.newaxis]

        logPlOld = np.array([0])

    else:

        logtmp = logsumexp(logsumexp(logPhSwitchmodelLast + logPhOldLast[np.newaxis, np.newaxis, :, :], axis=2) + logPlOldLast[np.newaxis, np.newaxis, :], axis=2)

        
        # logPhOld = logPchLast[:, np.newaxis] + logtmp
        logPhOld = logtmp
        norm = logsumexp(logPhOld, axis=0)
        if np.sum(np.isinf(norm)) == 0:
            logPhOld = logPhOld - norm[np.newaxis, :]
        else:
            for i in range(norm.shape[0]):
                if not np.isinf(norm[i]):
                    logPhOld[:, i] = logPhOld[:, i] - norm[np.newaxis, i]


        logPlOld = logsumexp(logPchLast[:, np.newaxis] + logtmp, axis=0) + logsumexp(logPlRunlengthmodelLast + logPlOldLast[np.newaxis, :], axis=1)
        norm = logsumexp(logPlOld)
        if not np.isinf(norm):
            logPlOld = logPlOld - norm
        logPlOld = np.atleast_1d(logPlOld)  # deal with the situation when PlOld turns out to be a scalar

    return logPhOld, logPlOld
def calculate_logPh(logPhSwitchmodel, logPhOld, logPlRunlengthmodel, logPlOld):
    logPh = logsumexp(logsumexp(logsumexp(logPhSwitchmodel + logPhOld[np.newaxis, np.newaxis, :, :], axis=2) + logPlRunlengthmodel[np.newaxis, :] + logPlOld[np.newaxis, np.newaxis, :], axis=2), axis=1)
    logPh = logPh - logsumexp(logPh)  # normalize to solve potential numerical deviation from sum to 1
    return logPh

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

    return matchedCount



# def trialLikelihood(t,hypotheses, iTrial, action, choiceIndex_valid,logPhOldLast, logPlOldLast,logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,rewardSetting = rewardSetting):
#     consistentCH = consistentCHAll[informed, numRD]

#     logPhSwitchmodel = logPhSwitchmodelAll[NHypos, t]


#     if t == 0:  # the first trial (with response)
#         logp = np.log(hPrior)
        
#         # for lOld in range(t):
#         #     loglik = np.zeros(NHypos)
#         #     # if iTrial <1:
#         #     for iH, h in enumerate(hypotheses):
#         #         pReward = rewardSetting[np.sum(~np.isnan(h)) - 1][np.sum(np.equal(h, stimuliThisGame[iTrial - 1 - lOld]))]
#         #         loglik[iH] = np.log(pReward) if rewardThisGame[iTrial - 1 - lOld] else np.log(1 - pReward)
#         #     logp = loglik + logp
#         #     logp = logp - logsumexp(logp)
#         #     posterior[iTrial, lOld] = np.exp(logp)

#         logPh = logPhSwitchmodel

#         # save for use on next trial - part1
#         logPhOld = logPh
#         logPhOldLast = logPhSwitchmodel

#         logPlOld = None
#         logPlOldLast = None
#         logPchLast = 0


#         # logPhSwitchmodel = None
#         logPlRunlengthmodel = None

#         logPhSwitchmodelLast = None
#         logPlRunlengthmodel = None
#         logPlRunlengthmodelLast = None
#         # choiceIndex_valid.append(hypotheses.index(action))  # save the index of the action chosen in this trial
#         # t += 1

#     else:
#         #更新先验
#         logp = np.log(hPrior)
#         posterior = dict.fromkeys([(iTrial, lOld) for lOld in range(iTrial)])
        
#         for lOld in range(trial):
#             loglik = np.zeros(NHypos)
#             for iH, h in enumerate(hypotheses):
#                 # column_idx = np.sum(np.equal(h, stimuliThisGame[iTrial - 1 - lOld]))
#                 row_idx = np.sum(~np.isnan(h)) - 1  # rewardSetting里第几行 -> #dim relevant
#                 column_idx = getColumnIdx(h, hypothesized_action=allHypothesizedActions[iH], action=action, row_idx=row_idx)  # rewardSetting里第几列 -> #similar elements  
#                 pReward = rewardSetting[row_idx][column_idx]
#                 loglik[iH] = np.log(pReward) if rewardThisGame[iTrial - 1 - lOld]>70 else np.log(1 - pReward)
#             logp = loglik + logp
#             logp = logp - logsumexp(logp)
#             posterior[trial, lOld] = np.exp(logp)
#         # logPhSwitchmodel,logPlRunlengthmodel
#         # logPhSwitchmodelLast = logPhSwitchmodel
#         # logPlRunlengthmodelLast = logPlRunlengthmodel

#         # recursive calculation (second and fourth terms)
#         # logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(t, iTrial, posterior, logPhOld, betaStay, thetaStay)
#         logPhOld, logPlOld = calculate_24terms(t, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)
#         logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(t,iTrial,  posterior, logPhOld, betaStay, thetaStay)
#         # logPlRunlengthmodelLast = logPlRunlengthmodel
#         logPlOldLast = logPlOld
#         logPh = calculate_logPh(logPhSwitchmodel, logPhOld, logPlRunlengthmodel, logPlOld)


#         logPhOldLast = logPhOld
#         logPlOldLast = logPlOld
#         logPhSwitchmodelLast = logPhSwitchmodel
#         logPlRunlengthmodelLast = logPlRunlengthmodel

#     logPchFull, logpChoice = choicePolicy_epsilon(NChoices, NHypos, logPh, epsilon)
#     # a = int(choiceIndex_valid[t])
#     logPchLast = logPchFull[int(choiceIndex_valid[t]), :]
#     logPhOld = logPh
#     t += 1
               
#     # logPchLast = logPchFull[int(choiceIndex[t-1]), :]

#     # likelihood of the trial
#     llh[iRow] = logpChoice[int(choiceIndex_valid[t-1])]
    
#     if returnPrediction:
#         samplePList.append(np.exp(logpChoice))

#     if returnPh:
#         PhList.append(np.exp(logPh))
    

#     i_valid = [True]*gameLength
#     if not (returnPrediction | returnTrialLikelihood | returnQvalues | returnPh):
#         results = -np.sum(llh[i_valid])
#     else:
#         results = []
#         results.append(-np.sum(llh[i_valid]))
#         if returnPrediction:
#             results.append(samplePList)
#         if returnTrialLikelihood:
#             results.append(llh)
#         if returnQvalues:
#             results.append(QfeatList)
#         if returnPh:
#             results.append(PhList)
#         results.append(logpChoice)


#     return t, results,logPhOldLast, logPlOldLast,logPhSwitchmodelLast,logPlRunlengthmodel,logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,logPhOld,logPlOld, logpChoice


#     # for iH, h in enumerate(hypotheses):
#     #     hypothesized_action = hypothesisToAction(h, options)

        




#     #     # row_idx = np.sum(~np.isnan(h))-1  # rewardSetting里第几行 -> #dim relevant
#     #     # column_idx = np.sum(np.equal(hypothesized_action, action)) 
#     #     # column_idx = getColumnIdx(h, hypothesized_action, action, row_idx) # rewardSetting里第几列 -> #similar elements
#     #     # pReward = rewardSetting[row_idx][column_idx]
#      #     # rewardWeight = reward/100
#     #     # lik[iH] = pReward if reward>70 else (1-pReward) # bayesian_data_3, falsely exclude the reward weight
#     #     # # lik[iH] = (pReward*rewardWeight) if reward>70 else pReward*0.2 # maybe bayesian_data_2
#     #     # lik[iH] = pReward * (reward/100) # bayesian_data_1, only this line
#     #     # if lik[iH] > max_lik:
#     #     #     max_lik = lik[iH]
#     #     #     similarity = column_idx
#     # return lik,similarity





# hypothesis testing: LRT or counting
def hypothesisTestingPolicy_LRTest(t,iTrial, posterior, logPhOld, betaStay, thetaStay):
    PlRunlengthmodel = np.zeros((t + 1, t))
    logPlRunlengthmodel = np.log(PlRunlengthmodel)
    for lOld in range(t):
        Ph_m = posterior[(iTrial,lOld)]
        LR = np.log(Ph_m / (1 - Ph_m))
        pStayCondH = 1 / (1 + np.exp(- betaStay * (LR - thetaStay)))
        logpStay = logsumexp(np.log(pStayCondH) + logPhOld[:, lOld])
        logpStay = 0 if logpStay > 0 else logpStay  # solve numerical issue
        logpSwitch = np.log(1 - np.exp(logpStay))
        [logpStayNormed, logpSwitchNormed] = [logpStay, logpSwitch] - logsumexp([logpStay, logpSwitch])
        logPlRunlengthmodel[lOld + 1, lOld] = logpStayNormed
        logPlRunlengthmodel[0, lOld] = logpSwitchNormed
    return logPlRunlengthmodel

# def hypothesisTestingPolicy_Counting(t, iTrial, estimatedPReward, logPhOld, betaStay, thetaStay):
#     PlRunlengthmodel = np.zeros((t + 1, t))
#     logPlRunlengthmodel = np.log(PlRunlengthmodel)
#     for lOld in range(t):
#         pStayCondH = 1 / (1 + np.exp(- betaStay * (estimatedPReward - thetaStay)))
#         logpStay = logsumexp(np.log(pStayCondH) + logPhOld[:, lOld])
#         logpStay = 0 if logpStay > 0 else logpStay  # solve numerical issue
#         logpSwitch = np.log(1 - np.exp(logpStay))
#         [logpStayNormed, logpSwitchNormed] = [logpStay, logpSwitch] - logsumexp([logpStay, logpSwitch])
#         logPlRunlengthmodel[lOld + 1, lOld] = logpStayNormed
#         logPlRunlengthmodel[0, lOld] = logpSwitchNormed
#     return logPlRunlengthmodel



# # hypothesis switching policy: random-switch or value-based
def calculate_logPhSwitchmodel_valueBasedNoReset(NHypos, t, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis):
    # calculate expected reward for all hypotheses based on Qfeat and costThis
    ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[t]) - costThis
    # consider cost per dimension
    
    # determine p(switch) for all hypotheses except for the currently tested one
    logpSwitchCached = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
    if t == 0:  # the first trial
        logpSwitch = deepcopy(logpSwitchCached)
        if betaTest is None: # models that always test
            logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
        else:
            pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
            logpSwitch[0] = np.log(1 - pTest)  # log softmax
            logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        logPhSwitchmodel = logpSwitch
    else:
        logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
        # stay
        for lOld in range(t):
            lNew = lOld + 1
            for iH in range(NHypos):
                logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1)
        # switch
        lNew = 0
        for iHOld in range(NHypos):
            logpSwitch = deepcopy(logpSwitchCached)
            if betaTest is None: # models that always test
                logpSwitch[iHOld] = np.log(0)
                logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
            else: # determine whether to test or not
                pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
                logpSwitch[0] = np.log(1 - pTest)  # log softmax
                if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                    logpSwitch[iHOld] = np.log(0)
                logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
            logPhSwitchmodel[:, lNew, iHOld, :t] = logpSwitch[:, np.newaxis]  # independent of run length
    return logPhSwitchmodel



# def calculate_logPhSwitchmodel_randomSwitch(NHypos, t, h_idx, loghPriorW, pTest):
#     if t == 0:
#         logpSwitch = deepcopy(loghPriorW)
#         if pTest is not None:
#             # 设置选择当前 h_idx 的概率为 1 - pTest
#             logpSwitch[h_idx] = np.log(1 - pTest)

#             # 剩下的 hypothesis 除了 h_idx，分摊 pTest 的部分
#             idx_others = [i for i in range(NHypos) if i != h_idx]
#             # logpSwitch[idx_others] = logpSwitch[idx_others] - logsumexp(logpSwitch[idx_others]) + np.log(pTest)
#             logpSwitch[idx_others] = np.log(pTest)
#         logPhSwitchmodel = logpSwitch  # shape: (NHypos,)
    
#     else:
#         logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
#         # stay：只允许 iH_new == iH_old，l_new = l_old + 1
#         for lOld in range(t):
#             lNew = lOld + 1
#             for iH in range(NHypos):
#                 # logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1)
#                 logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1 - pTest)
#         # switch：从 iH_old 切换到其他 hypothesis（不能回到自己）
#         for iHOld in range(NHypos):
#             logpSwitch = deepcopy(loghPriorW)
#             if pTest is None:
#                 # 强制切换，去掉自己
#                 logpSwitch[iHOld] = np.log(0)
#                 logpSwitch = logpSwitch - logsumexp(logpSwitch)
#             else:
#                 # 每次根据当前的 h_idx 分配 (1 - pTest) 的概率，其余为 pTest
#                 logpSwitch[h_idx] = np.log(1 - pTest)

#                 # 剩下的除了 h_idx 和 iHOld 的分摊 pTest
#                 idx_others = [i for i in range(NHypos) if i != h_idx and i != iHOld]
#                 # logpSwitch[idx_others] = logpSwitch[idx_others] - logsumexp(logpSwitch[idx_others]) + np.log(pTest)
#                 logpSwitch[idx_others] = np.log(pTest)
#                 # 强制不能回到 iHOld
#                 # logpSwitch[iHOld] = np.log(0)
#             # logPhSwitchmodel[new_h, l_new=0, old_h, l_old] = p(switch)
#             logPhSwitchmodel[:, 0, iHOld, :t] = logpSwitch[:, np.newaxis]

#     return logPhSwitchmodel
def calculate_logPhSwitchmodel_randomSwitch(NHypos, t, loghPriorW, pTest):
    if t == 0:  # the first trial
        logpSwitch = deepcopy(loghPriorW)
        
        # if pTest is not None: # determine whether to test or not
        #     b = 1
        #     logpSwitch[0] = np.log(1 - pTest)  # log softmax
        #     logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        logPhSwitchmodel = logpSwitch
    else:
        # logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
        logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), -np.inf)
        # stay
        for lOld in range(t):
            lNew = lOld + 1
            for iH in range(NHypos):
                # logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1)
                logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1 - pTest)
        # switch
        for iHOld in range(NHypos):
            logpSwitch = deepcopy(loghPriorW)  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
              # normalize
            # determine whether to test or not
                # logpSwitch[:] = logpSwitch[:] - logsumexp(logpSwitch) + np.log(pTest)
            # logpSwitch[iHOld] = np.log(1 - pTest)
            logpSwitch[iHOld] = -np.inf #防止switch到自己
            logpSwitch = logpSwitch - logsumexp(logpSwitch) + np.log(pTest)  # normalize and scale by pTest
            # for iHNew in range(NHypos):
            #     if iHNew == iHOld:
            #         continue
            #     logPhSwitchmodel[iHNew, 0, iHOld, :t] = logpSwitch[iHNew]
                # b = np.log(1 - pTest)
                # c = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)
                # logpSwitch[0] = np.log(1 - pTest)  # log softmax
                # if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                #     logpSwitch[iHOld] = np.log(0)
                # logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
            for iHNew in range(NHypos):
                if iHNew == iHOld:
                    continue
                logPhSwitchmodel[iHNew, 0, iHOld, :t] = logpSwitch[iHNew]
            # logPhSwitchmodel[:, 0, iHOld, :t] = logpSwitch[:, np.newaxis]
    return logPhSwitchmodel


# choice policy
# choice policy: (1) epsilon greedy: a special case of (2) with kChoice=0
# def choicePolicy_epsilon(NChoices, NHypos, logPh, consistentCH,epsilon):
#     logPchFull = np.log(epsilon / NChoices) * np.ones((NChoices, NHypos))
#     logPchFull[consistentCH] = np.log(1 - epsilon + epsilon / NChoices)

#     logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
#     logpChoice = logpChoice - logsumexp(logpChoice)

#     return logPchFull, logpChoice
def choicePolicy_epsilon(NChoices, NHypos, logPh, allHypothesizedActions, epsilon):
    """
    - hypo_to_action: 长度为 NHypos 的数组，每个元素是对应 hypothesis 下的 action index（整数）
    """

    # 基础：uniform noise over all actions
    pChoice = np.ones(NChoices) * (epsilon / NChoices)

    # 把每个 hypothesis 的信念加到它对应支持的 action 上
    p_h = np.exp(logPh)  # belief over hypotheses
    for h_idx in range(NHypos):
        # a_idx = allHypothesizedActions[h_idx]  # hypothesis h_idx 对应的 action
        pChoice[h_idx] += (1 - epsilon) * p_h[h_idx]

    # 转为 log 空间（用于后续 log likelihood）
    logpChoice = np.log(pChoice)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return None, logpChoice

# choice policy: (2) allowing for selecting more features than the hypothesis
def choicePolicy_selectMore(NChoices, NHypos, logPh, numMoreDim, epsilon, kChoice):
    kernel = np.exp(kChoice * numMoreDim)
    kernel[np.isnan(numMoreDim)] = 0
    logPchFull = np.log( kernel / np.nansum(kernel, axis=0) * (1 - epsilon) + epsilon / NChoices ) # probability for all the "compatible" choices sum to 1 - epsilon

    logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return logPchFull, logpChoice


# main function
if __name__=='__main__':
    for agent in range(100):
        # prev_p = {i: [] for i in range(0, 63)}
        reward = 0

        p = np.full(len(hypotheses), 1 / len(hypotheses)) # 初始化p的值

        phase = 'p1'
        nF = 3 
        if phase == 'p1':
            nD = 3
        else:
            nD = 4

        save_pth = 'E:/serial_hypothesis_1'
        save_pth = save_pth + '/'+str(agent)+'_fitting.csv'
        folder_path = os.path.dirname(save_pth)
        if not os.path.exists(folder_path):           # 检查文件夹是否存在
            os.makedirs(folder_path) 
            
 
        data_columns = ['trial', 'phase', 'action', 'reward','h_idx', 'h_action','logPh','logPlold','logPhSwitchmodel','logPLRunlengthmodel','posterior','logpChoice','loglik','logp','similarity'] 
        df = pd.DataFrame(columns = data_columns) 
        df.to_csv(save_pth, index=False)

        # get options from data files
        options_3D = pd.read_csv("E:\humans-combine-value-learning-and-hypothesis-testing-main\data\options_3_Dimension.csv")
        options_4D = pd.read_csv("E:\humans-combine-value-learning-and-hypothesis-testing-main\data\options_4_Dimension.csv")
# hi i'm here
        score_of_all_food = {'0':[0]*3,'1':[10,5,1],'2':[10,5,1],'3':[0]*3}

        draw_dict = {'agent_id':[],'round':[],'score':[],
                'choice_1':[],'choice_2':[],'choice_3':[],
                'pattern_1':[],'pattern_2':[],'pattern_3':[],
                'pos_1':[],'pos_2':[],'pos_3':[]}
        
        all_best_hypothesis_idx = []
        stimuliThisGame = []
        rewardThisGame = []
        choiceIndex_valid = []


        lik = np.zeros(len(hypotheses))
        similarity = 0
        max_lik = 0
        betaStay = 0.2
        thetaStay = 1
        if random_switch:
            pTest = 0.2 #初始的pSwitch
        else:
            pTest = 0.2
        epsilon = 0.3 #greedy
        consistentCHAll = dict.fromkeys(taskCondKeys)
        def getConsistentCH(allHypothesizedActions):
            n = len(allHypothesizedActions)
            output = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    h1 = allHypothesizedActions[i]
                    h2 = allHypothesizedActions[j]
                    # 判断是否每个位置都相等或都是 nan
                    match = np.all((np.isnan(h1) & np.isnan(h2)) | (h1 == h2))
                    output[i, j] = match
            return output

        # consistentCHAll[informed, numRD] = getConsistentCH(allHypothesesAll[informed, numRD])
        # kChoice = 
        cost = 0
        

        
        NChoices = len(hypotheses)
        NHypos = len(hypotheses)
        
        allHypothesesAll,NHyposAll,hPriorAll = emptyDicts(numDict=3,keys=[(False, 2)], lengthList=0)
        allHypothesesAll[informed, 2] = hypotheses
        hPriorAll[informed, 2] = np.ones(len(hypotheses))/len(hypotheses)
        NHyposAll[informed, 2] = NHypos
        consistentCHAll[informed, 2] = getConsistentCH(allHypothesesAll[informed, 2])


        hPrior = hPriorAll[False, 2]
        
        loghPriorWAll = dict(zip(taskCondKeys, [np.log(hPriorAll[key]) for key in taskCondKeys]))

        llh = np.zeros(gameLength)

        [returnPrediction ,returnQvalues, returnPh,returnTrialLikelihood] = [False]*3+[True]
        if returnPrediction:
            samplePList = []
        if returnQvalues:
            QfeatList = []
        if returnPh:
            PhList = []
        iRow = 0
        numRD = 2
        NHypos, loghPriorW = NHyposAll[informed, numRD], loghPriorWAll[informed, numRD]
        consistentCH = consistentCHAll[informed, numRD]
        t = 0
        Counting = True

                                
        # logPhSwitchmodel = None
        

        logp = np.log(hPrior)
        

        

        # save for use on next trial - part1
        # logPhOld = logPh

        
        logPchLast = 0
        # logPhOldLast, logPlOldLast,
        # logPhSwitchmodel,logPlRunlengthmodel
        h_idx = np.random.choice(np.arange(NChoices), size=1)[0]
        choiceIndex_valid.append(h_idx)
        for trial in range(30):
            if reward ==100:
                break
            else:
                if trial !=0:
                    max_value = np.max(logpChoice)
                    max_indices = np.where(logpChoice == max_value)[0]  # 返回所有最大值的索引
                    # # 从最大值索引中随机选择一个
                    h_idx = np.random.choice(max_indices)
                    # pChoice = np.exp(logpChoice)
                    # pChoice = pChoice / np.sum(pChoice)
                    # h_idx = np.random.choice(np.arange(NChoices), p = (pChoice),size=1)[0]

                # 记录这一轮更新后，用于下一轮的最好的hypothesis
                    llh[iRow] = logpChoice[h_idx]

                # save for use on next trial - part2
                # logPchLast = logPchFull[h_idx, :]
                    logPchLast = logpChoice
                options = get_options(trial, phase)
                options = [ast.literal_eval(x) for x in list(options)]
                allChoices = generate_choices(options)
                allHypothesizedActions = []
                for iH, h in enumerate(hypotheses):
                    hypothesized_action = hypothesisToAction(h, options)
                    allHypothesizedActions.append(hypothesized_action)
                action = hypothesisToAction(hypotheses[h_idx], options)
                # generate stimulus and reward outcome
                stimulus = deepcopy(action)

                #每一轮结束先单拎出来计算reward
                reward = rewardFunc(action)
                rewardThisGame.append(reward)  

                #每一轮包括第0轮结束后，用reward和action进入tllh函数得到更新，得到下一轮的选择依据（h_action）

                # t, result,logPhOldLast, logPlOldLast,logPhSwitchmodelLast,logPlRunlengthmodel,logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,logPhOld,logPlOld, logpChoice = trialLikelihood(t,hypotheses, trial,action, choiceIndex_valid,logPhOldLast, logPlOldLast, logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,rewardSetting = rewardSetting)
                iRow += 1
                choiceIndex_valid.append(h_idx)
                h_action = hypotheses[h_idx]
                stimuliThisGame.append(h_action)

            # # 得到对应的reward
                draw_dict = define_choice_pattern(phase,draw_dict,action)
            
            # reward = rewardFunc(action)
                draw_dict['score'].append(reward)
                  # 随机选择一个hypothesis
                # 
                # if trial == 0:
                    # logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
                

                draw_dict['agent_id'].append(agent)
                draw_dict['round'].append(trial)
                
                # 得到options
                
                logPhSwitchmodel = calculate_logPhSwitchmodel_randomSwitch(NHypos, trial, loghPriorW, pTest)

                if trial == 0:
                    # logPh = P(h_{t-1}|l_{t-1},c_{1:t-1})
                    logPl = None
                    logPlOldLast = None
                    # logPhSwitchmodel = None
                    logPhSwitchmodelLast = None
                    logPlRunlengthmodel = None
                    logPlRunlengthmodelLast = None
                    logPh = logPhSwitchmodel
                    # logPhOld = logPhSwitchmodel
                    logPhOldLast = logPhSwitchmodel

                    # logPl = P(l_{t-2}|c_{1:t-2})
                    logPlOld = None
                    loglik = np.zeros(NHypos)
                    col_list = np.zeros(NHypos)
                    logpChoice = np.zeros(NHypos)
                     
              
                else:
                    # logPhOld = logPh
                    
                    # action = hypothesisToAction(last_h_action, options) # 本轮的action由上一轮的最好的h得到
                    logPhOld, logPlOld = calculate_24terms(trial, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)
                    logp = np.log(hPrior)
                    posterior = dict.fromkeys([(trial, lOld) for lOld in range(trial)])
                    
                    for lOld in range(trial):
                        loglik = np.zeros(NHypos)
                        preward_list = np.zeros(NHypos)
                        col_list = np.zeros(NHypos)
                        for iH, h in enumerate(hypotheses):
                            # column_idx = np.sum(np.equal(h, stimuliThisGame[iTrial - 1 - lOld]))
                            row_idx = np.sum(~np.isnan(h)) - 1  # rewardSetting里第几行 -> #dim relevant
                            column_idx = getColumnIdx(h, hypothesized_action=allHypothesizedActions[iH], action=action, row_idx=row_idx)  # rewardSetting里第几列 -> #similar elements  
                            pReward = rewardSetting[row_idx][column_idx]
                            loglik[iH] = np.log(pReward) if rewardThisGame[trial - 1 - lOld]>70 else np.log(1 - pReward)

                            # reward_binary = int(rewardThisGame[trial - 1 - lOld] > 70)
                            # loglik[iH] = reward_binary * np.log(pReward) + (1 - reward_binary) * np.log(1 - pReward)
                            col_list[iH] = column_idx
                            
                            preward_list[iH] = np.log(pReward)

                        # iH = choiceIndex_valid[trial - 1 - lOld]  # 模拟中记录的
                        # row_idx = np.sum(~np.isnan(hypotheses[iH])) - 1
                        # column_idx = getColumnIdx(hypotheses[iH], hypothesized_action=allHypothesizedActions[iH], action=action, row_idx=row_idx)
                        # pReward = rewardSetting[row_idx][column_idx]
                        # loglik[iH] = np.log(pReward) if reward > 70 else np.log(1 - pReward)

                        logp = loglik + logp
                        logp = logp - logsumexp(logp)
                        posterior[trial, lOld] = np.exp(logp)
                    
                    logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(trial, trial, posterior, logPhOld, betaStay, thetaStay)
            # should get an action based on likelihood first, then update teh llh based on this reward and action
                    # last_h_idx = h_idx
                    # last_h_action = h_action

                    # logPhSwitchmodel = calculate_logPhSwitchmodel_randomSwitch(NHypos, trial, loghPriorW, pTest)

                    # posterior over hypotheses
                    a = 1
                    logPh = logsumexp(logsumexp(logsumexp(logPhSwitchmodel + logPhOld[np.newaxis, np.newaxis, :, :], axis=2) + logPlRunlengthmodel[np.newaxis, :] + logPlOld[np.newaxis, np.newaxis, :], axis=2), axis=1)
                    logPh = logPh - logsumexp(logPh)  # normalize to solve potential numerical deviation from sum to 1
                    Ph = np.exp(logPh)
                    # save for use on next trial - part1
                    logPhOldLast = logPhOld
                    logPlOldLast = logPlOld
                    logPhSwitchmodelLast = logPhSwitchmodel
                    logPlRunlengthmodelLast = logPlRunlengthmodel

                    logPchFull, logpChoice = choicePolicy_epsilon(NChoices, NHypos, logPh, allHypothesizedActions, epsilon)
            
                
  
            
            # 把reward和action记录下来
                row = {
                    "trial": trial+1,
                    "phase": phase,
                    "action": action,
                    "reward": reward,
                    # "likelihood_vector_for_h": str(llh),
                    'h_idx': h_idx,
                    'h_action': h_action,
                    'logPh':logPh,
                    # 'logPhOld':logPhOld,
                    'logPlold':logPlOld,
                    'logPhSwitchmodel':logPhSwitchmodel,
                    'logPLRunlengthmodel':logPlRunlengthmodelLast,
                    'posterior':posterior,
                    'logpChoice':logpChoice,
                    # 'logPchFull':logPchFull,
                    'loglik':loglik,
                    'logp': logp,
                    'similarity':col_list

                    # 'similarity':similarity
                    }
                df = pd.DataFrame([row])  # 将字典转为DataFrame
         
                df.to_csv(save_pth, mode="a", header=False, index=False)  # 追加写入文件

        pic_draw.choice_reward_pic('serial_hypothesis_1',nD,draw_dict)
            # pic_draw.choice_reward_pic('naive_RL_0615',nD,agent_dict)
            # draw_dict = define_choice_pattern(phase,draw_dict,act_data)