import numpy as np
import pandas as pd
import os
import scipy.stats
from itertools import combinations
from taskSetting_jilab import hypotheses_3D as hypotheses
from taskSetting_jilab import rewardSetting
import random
import ast
import pic_draw
from scipy.special import logsumexp
from copy import deepcopy
numFeaturesPerDimension = 3
informed = False
taskCondKeys = [(False, 2)]
numDimensions = 3
gameLength = 30
keys = [(trial, lOld) for trial in range(gameLength) for lOld in range(trial)]
posterior = dict(zip(keys, [None for _ in range(len(keys))]))
# data = 
choiceIndex_valid = []

def define_choice_pattern(phase,agent_dict,act_data): #把所有维度这一轮的选择模式加入列表

    for i in range(nD):
        dim_list = 'pattern_' +str(i+1)
        choice_seq_list = 'choice_' +str(i+1)
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

random_switch = False

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

        logPhOld = logPchLast + logPhOldLast
        norm = logsumexp(logPhOld)
        if not np.isinf(norm):
            logPhOld = logPhOld - norm
        logPhOld = logPhOld[:, np.newaxis]

        logPlOld = np.array([0])

    else:
        # print(logPhSwitchmodelLast.shape)
        # print(logPhOldLast.shape)
        # print()
        logtmp = logsumexp(logsumexp(logPhSwitchmodelLast + logPhOldLast[np.newaxis, np.newaxis, :, :], axis=2) + logPlOldLast[np.newaxis, np.newaxis, :], axis=2)

        
        logPhOld = logPchLast[:, np.newaxis] + logtmp
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
    # print(filteredAction,'\n')
    print('this is the trial number', trial)
    # print('this is the row index', row_idx)
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
    # print('matchedCount:', matchedCount)
    return matchedCount



# def trialLikelihood(t,hypotheses, iTrial, action, choiceIndex_valid,logPhOldLast, logPlOldLast,logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,rewardSetting = rewardSetting):
#     consistentCH = consistentCHAll[informed, numRD]

#     logPhSwitchmodel = logPhSwitchmodelAll[NHypos, t]
#     # logPhSwitchmodel = calculate_logPhSwitchmodel_randomSwitch(NHypos, iTrial, loghPriorW, pTest)
#     # print(logPhSwitchmodelAll[NHypos, t].shape)

#     # posterior = hPrior
    

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
#         posterior = dict.fromkeys([(trial, lOld) for lOld in range(trial)])
        
#         for lOld in range(trial):
#             loglik = np.zeros(NHypos)
#             for iH, h in enumerate(hypotheses):
#                 # column_idx = np.sum(np.equal(h, stimuliThisGame[iTrial - 1 - lOld]))
#                 row_idx = np.sum(~np.isnan(h)) - 1  # rewardSetting里第几行 -> #dim relevant
#                 column_idx = getColumnIdx(h, hypothesized_action=allHypothesizedActions[iH], action=action, row_idx=row_idx)  # rewardSetting里第几列 -> #similar elements  
#                 pReward = rewardSetting[row_idx][column_idx]
#                 loglik[iH] = np.log(pReward) if rewardThisGame[trial - 1 - lOld]>70 else np.log(1 - pReward)
#             logp = loglik + logp
#             logp = logp - logsumexp(logp)
#             posterior[trial, lOld] = np.exp(logp)
#         # logPhSwitchmodel,logPlRunlengthmodel
#         # logPhSwitchmodelLast = logPhSwitchmodel
#         # logPlRunlengthmodelLast = logPlRunlengthmodel

#         # recursive calculation (second and fourth terms)
#         # logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(t, iTrial, posterior, logPhOld, betaStay, thetaStay)
#         logPhOld, logPlOld = calculate_24terms(t, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)
#         logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(t,trial,  posterior, logPhOld, betaStay, thetaStay)
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

#     # print(h_idx, h_action)
#     return t, results,logPhOldLast, logPlOldLast,logPhSwitchmodelLast,logPlRunlengthmodel,logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,logPhOld,logPlOld, logpChoice


#     # for iH, h in enumerate(hypotheses):
#     #     hypothesized_action = hypothesisToAction(h, options)

        




#     #     # row_idx = np.sum(~np.isnan(h))-1  # rewardSetting里第几行 -> #dim relevant
#     #     # column_idx = np.sum(np.equal(hypothesized_action, action)) 
#     #     # column_idx = getColumnIdx(h, hypothesized_action, action, row_idx) # rewardSetting里第几列 -> #similar elements
#     #     # pReward = rewardSetting[row_idx][column_idx]
#     #     # print(row_idx, column_idx, pReward,'\n', h, '\n', hypothesized_action, '\n', action,'\n---------------------------------------------------\n')
#     #     # rewardWeight = reward/100
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
        # print(posterior)
        # print('ooooooooooooooooooo')
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

def calculate_logPhSwitchmodel_valueBasedReset(NHypos, t, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis):
    if t == 0:  # the first trial
        ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[0, 0]) - costThis # calculate expected reward for all hypotheses based on Qfeat and costThis
        logpSwitch = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
        # if betaTest is None: # models that always test
        #     logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
        # else:
        #     pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
        #     logpSwitch[0] = np.log(1 - pTest)  # log softmax
        #     logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
        logpSwitch = logpSwitch - logsumexp(logpSwitch) + np.log(pTest)  # normalize to pTest
        logPhSwitchmodel = logpSwitch
    else:
        ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[0, 0]) - costThis
        pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
        logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), -np.inf)
        # stay
        for lOld in range(t):
            lNew = lOld + 1
            for iH in range(NHypos):
                logPhSwitchmodel[iH, lNew, iH, lOld] = np.log(1-pTest)
        # switch
        lNew = 0
        for iHOld in range(NHypos):
            for lOld in range(t):
                # determine p(switch) for all hypotheses except for the currently tested one
                ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[t, lOld]) # calculate expected reward for all hypotheses based on Qfeat
                logpSwitch = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space

                # determine whether to test or not
                pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
                # logpSwitch[0] = np.log(1 - pTest)  # log softmax
                # if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                #     logpSwitch[iHOld] = np.log(0)
                logpSwitch = logpSwitch - logsumexp(logpSwitch) + np.log(pTest)  # normalize to pTest
                logPhSwitchmodel[:, lNew, iHOld, lOld] = logpSwitch
    return logPhSwitchmodel


# choice policy
# choice policy: (1) epsilon greedy: a special case of (2) with kChoice=0

def getNumMoreDim(allHypotheses,allChoices):
    compatibleChoice = np.array([[np.sum([np.isnan(h[i]) | (h[i] == c[i]) for i in range(numDimensions)]) == numDimensions for h in allHypotheses] for c in allChoices])
    numDiffDim = np.array([[len(c) - np.sum([(np.isnan(c[i]) & np.isnan(h[i])) | (c[i] == h[i]) for i in range(numDimensions)]) for h in allHypotheses] for c in allChoices])
    numMoreDim = np.empty(compatibleChoice.shape)
    numMoreDim[:] = np.nan
    numMoreDim[compatibleChoice] = numDiffDim[compatibleChoice]
    return numMoreDim



# choice policy: (2) allowing for selecting more features than the hypothesis
def choicePolicy_selectMore(NChoices, logPh, numMoreDim, epsilon, kChoice):
    kernel = np.exp(kChoice * numMoreDim)
    kernel[np.isnan(numMoreDim)] = 0
    logPchFull = np.log( kernel / np.nansum(kernel, axis=0) * (1 - epsilon) + epsilon / NChoices ) # probability for all the "compatible" choices sum to 1 - epsilon

    logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return logPchFull, logpChoice


def hypothesisToFeatureMat(hypotheses):
    featureMat = np.zeros([len(hypotheses), numDimensions * numFeaturesPerDimension])
    for iHypothesis, hypothesis in enumerate(hypotheses):
        for iDim in range(numDimensions):
            if np.isnan(hypothesis[iDim]):  # code nan as 3
                featureMat[iHypothesis, (iDim * numFeaturesPerDimension):((iDim + 1) * numFeaturesPerDimension)] = 1 / numFeaturesPerDimension
            else:
                featureMat[iHypothesis, int(iDim * numFeaturesPerDimension + hypothesis[iDim])] = 1
    return featureMat

def choiceToFeatureMat(choices):
    featureMat = np.zeros([len(choices), numDimensions * 3])
    for iChoice, choice in enumerate(choices):
        for iDim in range(numDimensions):
            if np.isnan(choice[iDim]):  # code nan as 3
                featureMat[iChoice, (iDim * 3):((iDim + 1) * 3)] = 1 / 3
            else:
                featureMat[iChoice, iDim * 3 + int(choice[iDim])] = 1
    return featureMat

def stimulusToFeatureMat(stimuli):
    featureMat = np.zeros([len(stimuli), numDimensions * 3])
    for iStimulus, stimulus in enumerate(stimuli):
        for iDim in range(numDimensions):
            featureMat[iStimulus, iDim * 3 + int(stimulus[iDim])] = 1
    return featureMat


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

        save_pth = 'E:/serial_hypothesis_4'
        save_pth = save_pth + '/'+str(agent)+'_fitting.csv'
        folder_path = os.path.dirname(save_pth)
        if not os.path.exists(folder_path):           # 检查文件夹是否存在
            os.makedirs(folder_path) 
            
        # print(save_pth)
        data_columns = ['trial', 'phase', 'action', 'reward', 'likelihood_vector_for_h','h_idx', 'logPh','logPlold','logPhSwitchmodel','logPLRunlengthmodel','posterior','logpChoice','loglik','logp','similarity'] 
        df = pd.DataFrame(columns = data_columns) 
        df.to_csv(save_pth, index=False)

        # get options from data files
        options_3D = pd.read_csv("E:\humans-combine-value-learning-and-hypothesis-testing-main\data\options_3_Dimension.csv")
        options_4D = pd.read_csv("E:\humans-combine-value-learning-and-hypothesis-testing-main\data\options_4_Dimension.csv")
# hi i'm here
        score_of_all_food = {'0':[10,5,1],'1':[10,5,1],'2':[0]*3,'3':[0]*3}

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
        betaStay = 0.5
        thetaStay = 0.5

        epsilon = 0.3 #greedy
        kChoice = 1 # number of features to select in the choice policy
        decay = 0.5
        eta = 0.5
        eta_r,eta_s = eta,eta
        betaSwitch = 0.5
        betaTest = 0.5
        thetaTest=0.5 
        costThis = 0.5 # cost of testing

        consistentCHAll = dict.fromkeys(taskCondKeys)
        def getConsistentCH(allHypothesizedActions):
            # return np.array([[np.sum([(np.isnan(c[i]) & np.isnan(h[i])) | (c[i] == h[i]) for i in range(numDimensions)]) == numDimensions for h in allHypothesizedActions] for c in allHypothesizedActions])
            # def getConsistentCH(hypotheses):
            # hypotheses: shape = (63, 3, 3)
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
        # consistentCHAll[informed, 2] = getConsistentCH(allHypothesesAll[informed, 2])


        hPrior = hPriorAll[False, 2]
        
        loghPriorWAll = dict(zip(taskCondKeys, [np.log(hPriorAll[key]) for key in taskCondKeys]))
        featureMatAllHypothesesAll = dict.fromkeys(taskCondKeys)
        for informed in [False]:
            for numRD in [2]:
                allHypotheses = allHypothesesAll[informed, numRD]
                allHypotheses_converted = np.zeros((len(allHypotheses), 3))
                for iHypothesis, hypothesis in enumerate(allHypotheses):
                    this_hypothesis = []
                    for iDim in range(numDimensions):
                        this_dim_hypothesis = hypothesis[iDim]
                        result = next((i for i, x in enumerate(this_dim_hypothesis) if not np.isnan(x)), np.nan)
                        this_hypothesis.append(result)
                    allHypotheses_converted[iHypothesis] = this_hypothesis
                featureMatAllHypothesesAll[informed, numRD] = hypothesisToFeatureMat(hypotheses=allHypotheses_converted)
        featureMatAllHypotheses = featureMatAllHypothesesAll[informed, 2]
        
        
        # logPhSwitchmodelAll = emptyDicts(numDict=1, keys=[], lengthList=0)
        # for numRD in [2]:
        NHypos, loghPriorW = NHyposAll[informed, 2], loghPriorWAll[informed, 2]
        # for x in range(gameLength):
        #     logPhSwitchmodelAll[(NHypos, x)] = calculate_logPhSwitchmodel_randomSwitch(NHypos, x, loghPriorW, pTest)

        # a = logPhSwitchmodelAll[(63,1)].shape
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
        choicesThisGame = []
        choicesThisGame_full = []
        # logPhOldLast, logPlOldLast,
        # logPhSwitchmodel,logPlRunlengthmodel
        h_idx = np.random.choice(np.arange(NChoices), size=1)[0]
        choiceIndex_valid.append(h_idx)

        

        keys = [(trial, lOld) for trial in range(gameLength) for lOld in range(trial+1)]
        Q0 = np.zeros(numDimensions * 3)
        Qfeat = dict(zip(keys, [Q0 for _ in range(len(keys))]))
        for trial in range(30):
            if reward ==100:
                break
            else:
                if trial > 0:
                    featureMatChoice = choiceToFeatureMat([choicesThisGame[trial - 1]])[0]
                    featureMatStimulus = stimulusToFeatureMat([stimuliThisGame[trial - 1]])[0]
                # h_action = hypotheses[h_idx]  # 随机选择一个hypothesis
                # 
                # if trial == 0:
                    # logPhSwitchmodel = np.full((NHypos, t + 1, NHypos, t), np.log(0))
                # logPhSwitchmodel = calculate_logPhSwitchmodel_randomSwitch(NHypos, trial, h_idx, loghPriorW, pTest)
                for lOld in range(trial):
                    if lOld == 0:
                        Qfeat[trial, lOld] = ((1 - decay) * featureMatStimulus + decay) * Q0 + (featureMatStimulus * eta_r + featureMatChoice * (eta_s - eta_r)) * (rewardThisGame[trial - 1] - np.dot(featureMatStimulus, Q0))
                    else:
                        Qfeat[trial, lOld] = ((1 - decay) * featureMatStimulus + decay) * Qfeat[trial - 1, lOld - 1] + (featureMatStimulus * eta_r + featureMatChoice * (eta_s - eta_r)) * (rewardThisGame[trial - 1] - np.dot(featureMatStimulus, Qfeat[trial - 1, lOld - 1]))
                logPhSwitchmodel = calculate_logPhSwitchmodel_valueBasedReset(NHypos, trial, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis)

                draw_dict['agent_id'].append(agent)
                draw_dict['round'].append(trial)
                


                # 得到options
                options = get_options(trial, phase)
                options = [ast.literal_eval(x) for x in list(options)]
                allChoices_Full = generate_choices(options)
                allChoices = allHypotheses_converted

                if trial != 0:
                    max_value = np.max(logpChoice)
                    max_indices = np.where(logpChoice == max_value)[0]  # 返回所有最大值的索引
                    # # 从最大值索引中随机选择一个
                    h_idx = np.random.choice(max_indices)
                    # h_action = hypotheses[h_idx]
                    # choice = allChoices[h_idx]

                if not np.isnan(allHypotheses_converted[h_idx]).any():
                    stimuliThisGame.append(allHypotheses_converted[h_idx])
                    action = hypothesisToAction(allHypotheses[h_idx], options)
                else:
                    this_hypothesis = allHypotheses_converted[h_idx]
                    isnan = np.isnan(allHypotheses_converted[h_idx])
                    this_hypothesis[isnan] = np.random.choice([0,1,2], size=isnan.sum())
                # stimulus = deepcopy(action)
                    stimuliThisGame.append(this_hypothesis)
                    full_hypo = np.full((numDimensions, numFeaturesPerDimension), np.nan)
                    this_hypothesis = np.array(this_hypothesis)
                    for dim in range(this_hypothesis.shape[0]):
                        full_hypo[dim][int(this_hypothesis[dim])] = 1
                    action = hypothesisToAction(full_hypo, options) 
                choicesThisGame.append(allHypotheses_converted[h_idx])
                #每一轮结束先单拎出来计算reward
                reward = rewardFunc(action)
                rewardThisGame.append(reward)

                allHypothesizedActions = []
                for iH, h in enumerate(hypotheses):
                    if iH == h_idx:
                        allHypothesizedActions.append(action)
                    else:
                        hypothesized_action = hypothesisToAction(h, options)
                        allHypothesizedActions.append(hypothesized_action)
                numMoreDim = getNumMoreDim(allHypotheses_converted,allHypotheses_converted)

                if trial == 0:
                    logPh = logPhSwitchmodel
                    logPhOldLast = logPhSwitchmodel
                    logPlOldLast = None
                    logPhSwitchmodelLast = None
                    # logPlRunlengthmodel = None
                    logPlRunlengthmodelLast = None
                    logPlOld = None
                    loglik = np.zeros(NHypos)
                    col_list = np.zeros(NHypos)
                    logpChoice = np.zeros(NHypos)
                    logPchFull = np.zeros(NHypos)
                            # print(len(allChoices))
                    
              
                else:
                    # logPhOld = logPh
                    # h_idx = np.random.choice(np.arange(NChoices), size=1, p=np.exp(logpChoice))[0]
                    

                    # action = hypothesisToAction(last_h_action, options) # 本轮的action由上一轮的最好的h得到
                    logPhOld, logPlOld = calculate_24terms(trial, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)
                    # logPhSwitchmodel = calculate_logPhSwitchmodel_randomSwitch(NHypos, trial, loghPriorW, pTest)
                    logp = np.log(hPrior)
                    posterior = dict.fromkeys([(trial, lOld) for lOld in range(trial)])
                    
                    for lOld in range(trial):
                        loglik = np.zeros(NHypos)
                        col_list = np.zeros(NHypos)

                        for iH, h in enumerate(hypotheses):
                            # column_idx = np.sum(np.equal(h, stimuliThisGame[iTrial - 1 - lOld]))
                            row_idx = np.sum(~np.isnan(h)) - 1  # rewardSetting里第几行 -> #dim relevant
                            column_idx = getColumnIdx(h, hypothesized_action=allHypothesizedActions[iH], action=action, row_idx=row_idx)  # rewardSetting里第几列 -> #similar elements  
                            pReward = rewardSetting[row_idx][column_idx]
                            loglik[iH] = np.log(pReward) if rewardThisGame[trial - 1 - lOld]>70 else np.log(1 - pReward)
                            col_list[iH] = column_idx
                        logp = loglik + logp
                        logp = logp - logsumexp(logp)
                        posterior[trial, lOld] = np.exp(logp)
                    
                    logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(trial, trial, posterior, logPhOld, betaStay, thetaStay)
            # should get an action based on likelihood first, then update teh llh based on this reward and action
                    # last_h_idx = h_idx
                    # last_h_action = h_action


                    # posterior over hypotheses
                    logPh = logsumexp(logsumexp(logsumexp(logPhSwitchmodel + logPhOld[np.newaxis, np.newaxis, :, :], axis=2) + logPlRunlengthmodel[np.newaxis, :] + logPlOld[np.newaxis, np.newaxis, :], axis=2), axis=1)
                    logPh = logPh - logsumexp(logPh)  # normalize to solve potential numerical deviation from sum to 1
                    # Ph = np.exp(logPh)
                    # save for use on next trial - part1
                    logPhOldLast = logPhOld
                    logPlOldLast = logPlOld
                    logPhSwitchmodelLast = logPhSwitchmodel
                    logPlRunlengthmodelLast = logPlRunlengthmodel

                logPchFull, logpChoice = choicePolicy_selectMore(NChoices, logPh, numMoreDim,epsilon,kChoice)
                # 记录这一轮更新后，用于下一轮的最好的hypothesis
                llh[iRow] = logpChoice[h_idx]
                # a = logPchFull 
                # save for use on next trial - part2
                logPchLast = logPchFull[h_idx, :]
                # h_idx = np.random.choice(len(hypotheses))
                  # 随机选择一个hypothesis
                # choiceIndex_valid.append(h_idx) # 存一个index用于第0轮的填空，从第1轮开始记录的是本轮的最好index
                # indChoice = np.random.choice(np.arange(NChoices), size=1, p=np.exp(logpChoice))[0]
                # choice = allChoices[indChoice]
                # choicesThisGame.append(choice)

                

                # max_value = np.max(logpChoice)
                # max_indices = np.where(logpChoice == max_value)[0]  # 返回所有最大值的索引

                # # 从最大值索引中随机选择一个
                # h_idx = np.random.choice(max_indices)
                # h_action = hypotheses[h_idx]
                # switch = np.random.choice([0, 1], p=[1 - pTest, pTest])
                  

                #每一轮包括第0轮结束后，用reward和action进入tllh函数得到更新，得到下一轮的选择依据（h_action）

                # t, result,logPhOldLast, logPlOldLast,logPhSwitchmodelLast,logPlRunlengthmodel,logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,logPhOld,logPlOld, logpChoice = trialLikelihood(t,hypotheses, trial,action, choiceIndex_valid,logPhOldLast, logPlOldLast, logPhSwitchmodelLast,logPlRunlengthmodelLast,logPchLast,rewardSetting = rewardSetting)
                iRow += 1
                choiceIndex_valid.append(h_idx)
                # real_action_full = hypothesisToAction(h_action, options)
                
                # generate stimulus and reward outcome
                
            # # 得到对应的reward
                draw_dict = define_choice_pattern(phase,draw_dict,action)
            
            # reward = rewardFunc(action)
                draw_dict['score'].append(reward)
            # print('this is the reward:\n',reward,'\n')
            
            # 把reward和action记录下来
                row = {
                    "trial": trial+1,
                    "phase": phase,
                    "action": action,
                    "reward": reward,
                    "likelihood_vector_for_h": str(llh),
                    'h_idx': h_idx,
                    # 'h_action': h_action,
                    # 'similarity':similarity,
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
                    }
                df = pd.DataFrame([row])  # 将字典转为DataFrame
                # print(save_pth)
                df.to_csv(save_pth, mode="a", header=False, index=False)  # 追加写入文件

        # print(all_best_hypothesis_idx)   # 每一轮最佳hypothesis的index
        # print(draw_dict)
        pic_draw.choice_reward_pic('serial_hypothesis_4',nD,draw_dict)
            # pic_draw.choice_reward_pic('naive_RL_0615',nD,agent_dict)
            # draw_dict = define_choice_pattern(phase,draw_dict,act_data)