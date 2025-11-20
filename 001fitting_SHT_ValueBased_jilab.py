import numpy as np
import pandas as pd
import os
import scipy.stats
from itertools import combinations
from funcsInferSerialHypoTesting import getCost
from taskSetting_jilab import hypotheses_3D, hypotheses_4D
from taskSetting_jilab import rewardSetting
import random
import ast
import pic_draw
from scipy.special import logsumexp, softmax
from copy import deepcopy
from utilities import *
from taskSetting_jilab import *




# task setting and load data
phase = 'p1'  # 'p1' with 3 dimensions or 'p2' with 4 dimensions
numFeaturesPerDimension = 3
informed = False
taskCondKeys = [(False, 2)]
model = 'SHT_ValueBased'
informed = False
taskCondKeys = [(False, 2)]
options_3D = pd.read_csv("C:\\Users\\DELL\\Desktop\\humans-combine-value-learning-and-hypothesis-testing-main\\data\\options_3_Dimension.csv")
options_4D = pd.read_csv("C:\\Users\\DELL\\Desktop\\humans-combine-value-learning-and-hypothesis-testing-main\\data\\options_4_Dimension.csv")
choices = {}
stimuli = {}



# flatten action_grid and turn into a feature matrix for calculation of reward
def stimulusToFeatureMat(action_grid, numDimensions, numFeaturesPerDimension):
    featureMat = np.zeros(numDimensions * numFeaturesPerDimension)
    row_weight = [10, 5, 1]

    action_grid_1 = [ast.literal_eval(x) for x in action_grid]

    # each 3 in a group
    action_grid_1 = [action_grid_1[i:i+3] for i in range(0, len(action_grid_1), 3)]

    food_score = {k: [0, 0, 0] for k in range(numDimensions)}
    for row_idx, row in enumerate(action_grid_1):
        for item in row:
            for dim_idx, feature_val in enumerate(item):
                food_score[dim_idx][int(feature_val - 1)] += row_weight[row_idx]
    for dim_idx, scores in food_score.items():
        for feature_idx, score in enumerate(scores):
            mat_idx = dim_idx * numFeaturesPerDimension + feature_idx
            featureMat[mat_idx] = score
    norm = np.linalg.norm(featureMat)
    norm=1
    if norm > 0: featureMat = featureMat / norm
    return featureMat

# transform hypotheses into feature matrix for calculation of expected reward
def hypothesisToFeatureMat(hypotheses_list, numDimensions, numFeaturesPerDimension):
    featureMat = np.zeros([len(hypotheses_list), numDimensions * numFeaturesPerDimension])
    for i_h, h in enumerate(hypotheses_list):
        care_ls=[]
        for i_dim in range(numDimensions):
            active_feature_idx = next((i for i, x in enumerate(h[i_dim]) if not np.isnan(x)), -1)
            if active_feature_idx != -1:
                mat_idx = i_dim * numFeaturesPerDimension + active_feature_idx
                featureMat[i_h, mat_idx] = 1
                care_ls.append(mat_idx)
    return featureMat

# transform action into feature matrix for calculation of expected reward, kind of like stimulusToFeatureMat
def choiceToFeatureMat(action_grid, numDimensions, numFeaturesPerDimension):
    featureMat = np.zeros(numDimensions * numFeaturesPerDimension)
    row_weight = [10, 5, 1]

    action_grid_1 = action_grid

    food_score = {k: [0, 0, 0] for k in range(numDimensions)}

    for row_idx, row in enumerate(action_grid_1):
        for item in row:
            
            for dim_idx, feature_val in enumerate(item):
                food_score[dim_idx][int(feature_val - 1)] += row_weight[row_idx]
    for dim_idx, scores in food_score.items():
        for feature_idx, score in enumerate(scores):
            mat_idx = dim_idx * numFeaturesPerDimension + feature_idx
            featureMat[mat_idx] = score
    norm = np.linalg.norm(featureMat)
    norm=1
    if norm > 0: featureMat = featureMat / norm
    return featureMat

def get_options(trial, phase):
    if phase == "P1":
        options = options_3D.iloc[:, trial].tolist()
    else:
        options = options_4D.iloc[:, trial].tolist()
    return options

def score_func(phase, option, h):
        if phase == 'P1':
            return np.nansum([h[i][int(option[i]) - 1] for i in range(3)])
        else:
            return np.nansum([h[i][int(option[i]) - 1] for i in range(4)])

# transform h into action under current options
def hypothesisToAction(phase, h, options):
    # h = [[1,nan,nan],[nan,nan,nan], [nan,nan,nan]]
    this_options = deepcopy(options)
    this_options = [ast.literal_eval(opt) for opt in this_options]
    random.shuffle(this_options)
    options_score_sort = sorted(this_options, key=lambda opt: score_func(phase, opt, h), reverse=True)
    # divide sorted lists into 3 groups, each group containing 3 lists -> get the hypothesized action
    options_score_sort = [options_score_sort[i:i + 3] for i in range(0, len(options_score_sort), 3)]
    return options_score_sort

def generate_choices_optimized(hypotheses,options,subject_choice=None):
    """
    Compare subject's action with h_action to find the corresponding index
    """
    allChoices = []
    seen = set()
    subject_choice_change = [subject_choice[i:i+3] for i in range(0, len(subject_choice), 3)]
    options_tuple = [tuple(opt) for opt in options]
    similarity_of_sub_and_h = []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    def compare_similarity(list1, list2, group_size=3):
        total_score = 0
        total_items = 0

        list1 = [x for group in list1 for x in group]
        list2_1 = [int("".join(map(str, sub))) for group in list2 for sub in group]
        for group1, group2 in zip(chunks(list1, group_size), chunks(list2_1, group_size)):

            set1 = set(group1)
            set2 = set(group2)

            intersection = set1.intersection(set2)
            score = len(intersection)

            total_score += score
            total_items += len(set1)

        return total_score

    for h in hypotheses:
        h_action = hypothesisToAction(phase,h, options)
        if subject_choice_change is not None:               

            similarity_of_this_h = compare_similarity(subject_choice_change, h_action)
            similarity_of_sub_and_h.append(similarity_of_this_h)

        h_action = [h_action[i:i+3] for i in range(0, len(h_action), 3)]
        h_action_list = [list(opt) for opt in h_action]
        allChoices.append(h_action_list)
    max_similarity_index = np.argmax(similarity_of_sub_and_h)

    def split_digits(num):
        """Split an integer num into a list of its digits, e.g., 332 -> [3,3,2]"""
        return [int(d) for d in str(num)]

    def transform(lst):
        """Split the three-digit numbers in a 2D list into digit lists"""
        return [[split_digits(num) for num in group] for group in lst]

    subject_choice_change = transform(subject_choice_change)
    allChoices[max_similarity_index] = subject_choice_change

    return max_similarity_index, allChoices



# ===== data preparation for model fitting: find the hypothesis mostly similar to participant's choices, then calculate posteriors =====
def prepForFitting_inferSerialHypoTesting(data):
    data = data.reset_index(drop=True)
    dataFitting = data.copy()
    dataFitting['All_hypothesis_action'] = None
    dataFitting['choiceIndex'] = None
    dataFitting['stimulus'] = None

    keys = ['dataFitting', 'logPhSwitchmodelAll','posterior','featureMatChoices', 'featureMatStimuli', 'featureMatAllHypothesesAll','numMoreDimAll','consistentCHAll','estimatedPReward']
    output = dict.fromkeys(keys) # initialize output dict for storing posteriors and feature matrices

    row_weight = [0.5, 0, -0.5]

    allFeaturesList = flatten2Dlist([DIMENSIONS_TO_FEATURES[dim] for dim in DIMENSIONS])
    
    choiceIndex, stimulusIndex = zerosLists(numList=2, lengthList=data.shape[0])
    choiceFeatureIndex, selectedFeature_order,stimulus_order,stimulusFeatureIndex, selectedFeature = emptyDicts(numDict=5, keys=DIMENSIONS, lengthList=data.shape[0])
    
    featureMatChoices, featureMatStimuli = [np.empty((data.shape[0], numDimensions * numFeaturesPerDimension)) for _ in range(2)]
    featureMatChoices[:] = None
    featureMatStimuli[:] = None
    for iRow in range(data.shape[0]):
        choice = []
        stim_seq = []
        actions = []
        row_1 = list(data.loc[iRow]['block1'])
        row_2 = list(data.loc[iRow]['block2'])
        row_3 = list(data.loc[iRow]['block3'])

        this_stim = row_1 + row_2 + row_3 # what participant actually chose
        stim_seq.append(this_stim)
        actions.append(this_stim)

        # find and record the hypothesis that best matches subject's choices
        max_similarity_index,All_hypothesis_action = generate_choices_optimized(hypotheses, get_options(dataFitting.loc[iRow, 'trial']-1, phase), subject_choice=this_stim)
        allChoices = All_hypothesis_action
        choices[iRow] = hypotheses[max_similarity_index] # the best matching hypothesis stored here

        All_hypothesis_action = {i+1: v for i, v in enumerate(All_hypothesis_action)}
        index = max_similarity_index
        dataFitting.loc[iRow, 'choiceIndex'] = index
        dataFitting.at[iRow, 'All_hypothesis_action'] = All_hypothesis_action
        
        # code stimulus as index according to allStimuli
        stimulus = get_options(dataFitting.loc[iRow, 'trial']-1, phase)
        stimuli[iRow] = stimulus
        dataFitting.at[iRow, 'stimulus'] = stimulus # what participant actually saw
        choiceIndex[iRow] = max_similarity_index
        dataFitting['choiceIndex']= choiceIndex
        output['dataFitting'] = dataFitting

        allHypothesesAll,NHyposAll,hPriorAll = emptyDicts(numDict=3,keys=[(False, 2)], lengthList=0)
        allHypothesesAll = hypotheses
        hPriorAll = np.ones(len(hypotheses))/len(hypotheses)
        keys = [(iTrial, lOld) for iTrial in range(gameLength) for lOld in range(iTrial)] # lOld = run-length (i.e., how many trials back from iTrial we look)
        posterior = dict(zip(keys, [None for _ in range(len(keys))])) # initialize posterior dict

        reward = dataFitting['reward'].values

        # hypothesis space
        allHypotheses = allHypothesesAll
        hPrior = hPriorAll
        NHypos = len(allHypotheses)
        t = 0
        round = int(dataFitting.loc[iRow, 'choiceIndex'])
        
        # >>> posterior calculation for all trials
        for iTrial in range(len(dataFitting)): 
            if t > 0:
                logp = np.log(hPrior)
                for lOld in range(t):
                    loglik = np.zeros(NHypos)
                    for iH, h in enumerate(allHypotheses):
                        col_idx = getColumnIdx(h, All_hypothesis_action[iH+1],allChoices[int(dataFitting.loc[t - 1 - lOld, 'choiceIndex'])], np.sum(~np.isnan(h)) - 1)
                        pReward = rewardSetting[np.sum(~np.isnan(h)) - 1][np.sum(col_idx)]
                        # here pReward is extracted from rewardSetting (find it in tasksetting file) according to how many relevant dimensions are in h and how many features in the choice match the relevant features in h
                        loglik[iH] = np.log(pReward) if reward[t - 1 - lOld]>70 else np.log(1 - pReward)
                    logp = loglik + logp
                    logp = loglik + logp
                    logp = logp - logsumexp(logp)
                    posterior[iTrial, lOld] = np.exp(logp)
            t += 1
        output['posterior'] = posterior

        # turn choices and stimuli into feature matrices
        featureMatChoices[iRow, :] = choiceToFeatureMat(allChoices[int(dataFitting.loc[iRow, 'choiceIndex'])], numDimensions, numFeaturesPerDimension)
        featureMatStimuli[iRow, :] = stimulusToFeatureMat(dataFitting.loc[iRow, 'stimulus'], numDimensions=numDimensions, numFeaturesPerDimension=numFeaturesPerDimension)
        featureMatAllHypothesesAll = dict.fromkeys(taskCondKeys)
        for numRD in np.arange(numDimensions) + 1:
            allHypotheses = allHypothesesAll
            featureMatAllHypothesesAll = hypothesisToFeatureMat(allHypotheses,numDimensions, numFeaturesPerDimension)
        output['featureMatAllHypothesesAll'] = featureMatAllHypothesesAll

    numMoreDimAll = dict.fromkeys(taskCondKeys)
    output['featureMatChoices'] = featureMatChoices
    output['featureMatStimuli'] = featureMatStimuli
    for numRD in np.arange(numDimensions) + 1:
        allHypotheses = allHypothesesAll[numRD]

    return output,choices,stimuli,allChoices 

def getColumnIdx(h, hypothesized_action,action, row_idx): # get the number of chosen stimuli in the action that match the hypothesized relevant features in h (i.e., the column index used in rewardsetting)
    # e.g, h = [[1,nan,nan],[nan,nan,nan], [nan,nan,nan]]
    if len(hypothesized_action) == 1:
        hypothesized_action = hypothesized_action[0]
    matchedCount = 0 # the column idx we want

    # Convert options without hypothesized relevant features into [0, 0, 0] so they’re ignored when comparing with the action; when comparing the attended hypothesis to action, only consider the features that are relevant
    def find_indices_with_1(h): # find the (dms, ftr) indices of all relevant (marked) value of h
        indices = []
        for i, row in enumerate(h):  
            for j, value in enumerate(row):
                if value == 1:
                    indices.append((i, j)) 
        return indices
    
    def count_matching_elements(hypothesized_action, indices):
        filteredAction = {}  # store the combinations and corresponding indices that meet the criteria for each sublist
        # iterate over each major sublist in hypothesized_action
        
        for group in hypothesized_action:
            for ha in group:  # iterate over each stimuli chosen in hypothesized action
                count = 0
                for i, j in indices:  # iterate over all (i, j) index pairs
                    ha_1 = ha
                    if ha_1[i] == j + 1:  # check if the i-th element equals j + 1
                        count += 1  # increment count if condition is met
                filteredAction[tuple(ha_1)] = count
        return filteredAction
    
    filteredAction = count_matching_elements(hypothesized_action, indices = find_indices_with_1(h))

    if len(action) == 1:
        action_1 = action[0]
    else:
        action_1 = action
    
    if row_idx==0: # for 1D relevant games
        for ha, count in filteredAction.items():
            ha_2 = list(ha)
            ha_2 = int("".join(map(str, ha_2)))
            if count==1:
                if ha_2 in action_1[0]:
                    matchedCount += 1
    elif row_idx==1:
        iterated_count = 0
        row_1_count = 0
        for ha, count in filteredAction.items():
            ha = list(ha)
            if count==2 and list(ha) in action_1[0]:
                matchedCount += 1
            elif count==1:
                # row_1_count += 1
                if list(ha) in action_1[0]:
                   if row_1_count<=2:
                       row_1_count += 1
                       matchedCount += 1
                elif list(ha) in action_1[1]:
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
                   if list(ha) in action_1[0]:
                        matchedCount += 1
                elif count==1:
                    if list(ha) in action_1[0]:
                        if row_1_count<2:
                            row_1_count += 1
                            matchedCount += 1
                    elif list(ha) in action_1[1]:
                        matchedCount += 1
        else:
            for ha, count in filteredAction.items():
                ha = list(ha)
                if count==2 and list(ha) in action_1[0]:
                    matchedCount += 1
                elif count==1 and list(ha) in action_1[1]:
                    matchedCount += 1
    return matchedCount

def pRewardMatrix(hypotheses, choices, rewardSetting=rewardSetting):
    pRewardMat = np.zeros((len(hypotheses), len(choices)))
    for iH, h in enumerate(hypotheses):
        for iC, choice in enumerate(choices):
            matchNum = 0
            numVisible = len(choice)  # usually 3
            for dim in range(3):  # 3 dimensions 
                stimulus = choice[dim]  # e.g., (1,2,3)
                # h[dim] is the preference ordering for that dimension, e.g., (1, 2, 3)
                if stimulus[0] == h[dim][0]:  # e.g., the first feature of stimulus is the top1 in this dimension's ordering
                    matchNum += 1
            pRewardMat[iH, iC] = rewardSetting[numVisible - 1][matchNum]
    return pRewardMat


parNames = ['betaStay','deltaStay','thetaStay',
            'eta_s', 'eta_r', 'decay', 'betaSwitch'
            'epsilon','kChoice','betaTest', 'thetaTest']

def choicePolicy_selectMore(NChoices, NHypos, logPh, numMoreDim, epsilon, kChoice):
    kernel = np.exp(kChoice * numMoreDim)
    kernel[np.isnan(numMoreDim)] = 0
    logPchFull = np.log( kernel / np.nansum(kernel, axis=0) * (1 - epsilon) + epsilon / NChoices ) # probability for all the "compatible" choices sum to 1 - epsilon

    logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return logPchFull, logpChoice

def choicePolicy_epsilon(NChoices, NHypos, logPh, epsilon):
    logPchFull = np.log(epsilon / NChoices) * np.ones((NChoices, NHypos))
    # logPchFull[consistentCH] = np.log(1 - epsilon + epsilon / NChoices)
    logpChoice = logsumexp(logPchFull + logPh[np.newaxis, :], axis=1)
    logpChoice = logpChoice - logsumexp(logpChoice)

    return logPchFull, logpChoice

def QfeatReset(featureMatChoices, featureMatStimuli, reward, eta_s, eta_r, decay):
    featureMatStimuli = featureMatStimuli / np.max(np.abs(featureMatStimuli))
    featureMatChoices = featureMatChoices / np.max(np.abs(featureMatChoices))

    keys = [(0, 0)] + [(t, lOld) for t in range(len(reward)) for lOld in range(t)]
    Q0 = np.zeros(numDimensions*numFeaturesPerDimension)
    
    Qfeat = dict(zip(keys, [Q0 for _ in range(len(keys))]))
    for t_start in range(len(reward) - 1):
        for lOld in range(len(reward) - t_start - 1):
            t = t_start + lOld + 1
            if lOld == 0:
                Qfeat[t, lOld] = ((1 - decay) * featureMatStimuli[t - 1, :] + decay) * Q0 + (featureMatStimuli[t - 1, :] * eta_r + featureMatChoices[t - 1, :] * (eta_s - eta_r)) * (reward[t - 1] - np.dot(featureMatStimuli[t - 1, :], Q0))
            else:
                # Qfeat[t, lOld] = ((1 - decay) * featureMatStimuli[t - 1, :] + decay) * Qfeat[t - 1, lOld - 1] + (featureMatStimuli[t - 1, :] * eta_r + featureMatChoices[t - 1, :] * (eta_s - eta_r)) * (reward[t - 1] - np.dot(featureMatStimuli[t - 1, :], Qfeat[t - 1, lOld - 1]))
                Qfeat[t, lOld] = (1 - decay) * Qfeat[t - 1, lOld - 1] + \
                                (featureMatStimuli[t - 1, :] * eta_r + featureMatChoices[t - 1, :] * (eta_s - eta_r)) * \
                                (reward[t - 1] - np.dot(featureMatStimuli[t - 1, :], Qfeat[t - 1, lOld - 1]))

    return Qfeat

def calculate_logPhSwitchmodel_valueBasedReset(NHypos, t, featureMatAllHypotheses, Qfeat, loghPriorW, betaSwitch, betaTest, thetaTest, costThis):
    if t == 0:  # the first trial
        ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[0, 0]) - costThis
        # print('Qfeat', Qfeat[0, 0])
         # calculate expected reward for all hypotheses based on Qfeat and costThis
        logpSwitch = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
        # if betaTest is None: # models that always test
        #     logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
        # else:
        #     pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
        #     logpSwitch[0] = np.log(1 - pTest)  # log softmax
        #     logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
        pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
        logpSwitch = logpSwitch - logsumexp(logpSwitch) + np.log(pTest)  # normalize to pTest
        # logPhSwitchmodel = logpSwitch
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
            for lOld in range(t):
                # determine p(switch) for all hypotheses except for the currently tested one
                ExpectedRHypo = np.dot(featureMatAllHypotheses, Qfeat[t, lOld]) # calculate expected reward for all hypotheses based on Qfeat
                logpSwitch = betaSwitch * ExpectedRHypo + loghPriorW  # multiplying probability by weight in the probability space is the same as adding log weight in the log space
                # if betaTest is None: # models that always test
                #     logpSwitch[iHOld] = np.log(0)
                #     logpSwitch = logpSwitch - logsumexp(logpSwitch)  # normalize to 1
                # else: # determine whether to test or not
                #     pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
                #     logpSwitch[0] = np.log(1 - pTest)  # log softmax
                #     if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                #         logpSwitch[iHOld] = np.log(0)
                #     logpSwitch[1:] = logpSwitch[1:] - logsumexp(logpSwitch[1:]) + np.log(pTest)  # normalize to pTest
                # logPhSwitchmodel[:, lNew, iHOld, lOld] = logpSwitch
                pTest = 1 / (1 + np.exp( - betaTest * (np.max(ExpectedRHypo) - thetaTest)))
                # logpSwitch[0] = np.log(1 - pTest)  # log softmax
                # if iHOld > 0: # can't switch to the old hypothesis, but only if it's not [np.nan, np.nan, np.nan]; otherwise, allow keep not testing
                #     logpSwitch[iHOld] = np.log(0)
                logpSwitch = logpSwitch - logsumexp(logpSwitch) + np.log(pTest)  # normalize to pTest
                logPhSwitchmodel[:, lNew, iHOld, lOld] = logpSwitch
                for iHNew in range(NHypos):
                    if iHNew == iHOld:
                        continue
                    logPhSwitchmodel[iHNew, 0, iHOld, :t] = logpSwitch[iHNew]
    # print('featureMatAllHypotheses',featureMatAllHypotheses)
    
    return logPhSwitchmodel



def calculate_24terms(t, logPhSwitchmodelLast=None, logPhOldLast=None, logPlOldLast=None, logPlRunlengthmodelLast=None, logPchLast=None):
    if t == 1:

        logPhOld = logPchLast + logPhOldLast
        norm = logsumexp(logPhOld)
        if not np.isinf(norm):
            logPhOld = logPhOld - norm
        logPhOld = logPhOld[:, np.newaxis]

        logPlOld = np.array([0])

    else:

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

def hypothesisTestingPolicy_LRTest(t, iTrial, posterior, logPhOld, betaStay, thetaStay):
    PlRunlengthmodel = np.zeros((t + 1, t))
    logPlRunlengthmodel = np.log(PlRunlengthmodel)
    for lOld in range(t):
        Ph_m = posterior[iTrial, lOld]
        LR = np.log(Ph_m / (1 - Ph_m))
        pStayCondH = 1 / (1 + np.exp(- betaStay * (LR - thetaStay)))
        logpStay = logsumexp(np.log(pStayCondH) + logPhOld[:, lOld])
        logpStay = 0 if logpStay > 0 else logpStay  # solve numerical issue
        logpSwitch = np.log(1 - np.exp(logpStay))
        [logpStayNormed, logpSwitchNormed] = [logpStay, logpSwitch] - logsumexp([logpStay, logpSwitch])
        logPlRunlengthmodel[lOld + 1, lOld] = logpStayNormed
        logPlRunlengthmodel[0, lOld] = logpSwitchNormed
    return logPlRunlengthmodel

def calculate_logPh(logPhSwitchmodel, logPhOld, logPlRunlengthmodel, logPlOld):
    logPh = logsumexp(logsumexp(logsumexp(logPhSwitchmodel + logPhOld[np.newaxis, np.newaxis, :, :], axis=2) + logPlRunlengthmodel[np.newaxis, :] + logPlOld[np.newaxis, np.newaxis, :], axis=2), axis=1)
    logPh = logPh - logsumexp(logPh)  # normalize to solve potential numerical deviation from sum to 1
    
    return logPh

def likelihood_inferSerialHypoTesting(allChoices,dataFitting,NChoices,NHyposAll):
    # load data prepared for fitting
    posterior = dataFitting['posterior']
    featureMatChoices = dataFitting['featureMatChoices']
    featureMatStimuli = dataFitting['featureMatStimuli']
    featureMatAllHypothesesAll = dataFitting['featureMatAllHypothesesAll']
    dataFitting = dataFitting['dataFitting']

    # parameters
    betaStay = 10
    thetaStay = 0.5
    eta_s = 0.05
    eta_r = 0.05
    decay = 0.1
    betaSwitch = 10
    epsilon = 0.02
    kChoice = 0.5
    betaTest, thetaTest = 800, 0.0046

    # initialize
    likelihood = 0
    optimal_prob_ls = 0
    random_prob_ls = 0
    results = {'phase':[],'subject':[], 'likelihood':[],'optimal_prob_ls':[],'random_prob_ls':[]}
    
    allHypothesesAll = hypotheses
    hPriorAll = np.ones(len(hypotheses))/len(hypotheses)
    loghPriorWAll = np.log(hPriorAll)
    costAll = getCost(cost, allHypothesesAll) # calculate cost, not used in this model version
    lik = np.zeros(len(hypotheses))
    llh = np.zeros(dataFitting.shape[0])
    iRow = 0
    choiceIndex = dataFitting['choiceIndex'].values
    reward = dataFitting['reward'].values
    featureMatChoices_valid = featureMatChoices[:, :]
    featureMatStimuli_valid = featureMatStimuli[:, :]
    numRD = 2

    # hypothesis space
    NHypos, loghPriorW = NHyposAll, loghPriorWAll
    featureMatAllHypotheses = featureMatAllHypothesesAll

    # get Q-values for value learning
    Qfeat_valid = QfeatReset(featureMatChoices_valid, featureMatStimuli_valid, reward=reward, eta_s=eta_s, eta_r=eta_r, decay=decay)
    
    # calculate likelihood across all trials
    t = 0
    costThis = 0.5
    for iTrial in range(len(dataFitting)):
        # calculate logPhSwitchmodel, i.e., the probability of switching to a new hypothesis based on value learning
        logPhSwitchmodel = calculate_logPhSwitchmodel_valueBasedReset(NHypos, t, featureMatAllHypotheses, Qfeat_valid, loghPriorW, betaSwitch, betaTest, thetaTest, costThis)
        if t == 0:  # the first trial (with response)
            logPh = logPhSwitchmodel

            # save for use on next trial - part1
            logPhOldLast = logPhSwitchmodel
            logPlOldLast = None
            logPhSwitchmodelLast = None
            logPlRunlengthmodelLast = None

        else:
            # recursive calculation (second and fourth terms)
            logPhOld, logPlOld = calculate_24terms(t, logPhSwitchmodelLast, logPhOldLast, logPlOldLast, logPlRunlengthmodelLast, logPchLast)
            logPlRunlengthmodel = hypothesisTestingPolicy_LRTest(t, iTrial, posterior, logPhOld, betaStay, thetaStay)
            # posterior over hypotheses
            logPh = calculate_logPh(logPhSwitchmodel, logPhOld, logPlRunlengthmodel, logPlOld)

            # save for use on next trial - part1
            logPhOldLast = logPhOld
            logPlOldLast = logPlOld
            logPhSwitchmodelLast = logPhSwitchmodel
            logPlRunlengthmodelLast = logPlRunlengthmodel
        

        logPchFull, logpChoice = choicePolicy_epsilon(NChoices, NHypos, logPh, epsilon)
        # save for use on next trial - part2
        logPchLast = logPchFull[int(choiceIndex[t]), :]
        ph = np.exp(logPh)
        pChoice = softmax(np.array(ph))
        pChoice = pChoice/np.sum(pChoice)
        likelihood = pChoice[int(choiceIndex[t])]

        opt_idx=np.argmax(pChoice)
        optimal_prob_ls = pChoice[opt_idx]
        random_choiceIndex = random.choice(range(len(pChoice)))
        random_prob_ls = pChoice[random_choiceIndex]

        results['phase'].append(dataFitting.loc[0,'phase'])
        results['subject'].append(dataFitting.loc[0,'subject'])
        results['likelihood'].append(likelihood)
        results['optimal_prob_ls'].append(optimal_prob_ls)
        results['random_prob_ls'].append(random_prob_ls)

        t+= 1
    iRow += 1
    return results

# ===== Main Script starts here =====
if __name__=='__main__':
    # load choice pattern data from csv and define dataframes to store processed data
    raw_df = pd.read_csv(r'C:\\Users\\DELL\\Desktop\\humans-combine-value-learning-and-hypothesis-testing-main\\data\\choice_category_uniform_105.csv')
    all_thing_df_p1 = {'subject':[],'trial':[],'phase':[],'dim':[],'reward':[],'sort':[],'block1':[],'block2':[],'block3':[]}
    all_thing_df_p2 = {'subject':[],'trial':[],'phase':[],'dim':[],'reward':[],'sort':[],'block1':[],'block2':[],'block3':[]}
    all_thing_df_p2only = {'subject':[],'trial':[],'phase':[],'dim':[],'reward':[],'sort':[],'block1':[],'block2':[],'block3':[]}
    dim_dict = {'P1':['dim0','dim1','dim2'],'P2':['dim0','dim1','dim2','dim3']}
    blocknum = 3
    all_df_dict = {'P1':all_thing_df_p1,'P2':all_thing_df_p2,'P2-only':all_thing_df_p2only}
    this_phase_df = all_df_dict[phase]

    # define dimensions, game length, hypotheses, etc. for each game phase
    for phase in ['P1','P2','P2-only']:
        if phase == 'P1':
            DIMENSIONS = [0, 1, 2]
            numDimensions = 3
            gameLength = 30
            DIMENSIONS_TO_FEATURES = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
            hypotheses = hypotheses_3D
            NChoices = 63 # of possible choices in 3D, which is actually number of hypotheses in this case
            NHyposAll = 63
        else:
            DIMENSIONS = [0, 1, 2, 3]
            numDimensions = 4
            gameLength = 50
            DIMENSIONS_TO_FEATURES = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}
            hypotheses = hypotheses_4D
            NChoices = 255
            NHyposAll = 255

        cost = 0.1 # not used in current model version

        if phase =='P1':
            dim_list = dim_dict['P1']
        else:
            dim_list = dim_dict['P2']

        for block in range(blocknum):
            for index, row in raw_df[raw_df['phase']==phase].iterrows():
                block_num = block + 1
                this_block_data = row['block'+str(block_num)]
                if block == 0:
                    this_phase_df['subject'].append(row['subject'])
                    this_phase_df['dim'].append(len(dim_list)) # number of relevant dimensions
                    this_phase_df['trial'].append(row['round'])
                    this_phase_df['reward'].append(row['score'])
                    this_phase_df['phase'].append(row['phase'])
                    this_round_choice = []
                    for game_row in range(3):
                        this_game_row_sort = list(ast.literal_eval(row['block'+str(game_row+1)]))
                        this_round_choice = this_round_choice + this_game_row_sort
                    this_phase_df['sort'].append(this_round_choice)
                this_phase_df['block'+str(block+1)].append(tuple(ast.literal_eval(this_block_data)))

    dict_of_data = {'phase':[], 'subject':[], 'likelihood':[],'optimal_prob_ls':[],'random_prob_ls':[]}
    all_data_df = pd.DataFrame(dict_of_data)


    for phase in ['P1','P2','P2-only']:
        all_df_dict[phase] = pd.DataFrame(all_df_dict[phase])
        datafitting = all_df_dict[phase]
        datafitting['choice'] = None

        # fitting starts here
        for subject in datafitting['subject'].unique():
            subject_data = datafitting[datafitting['subject'] == subject].reset_index(drop=True)
            subject_data,choices,stimuli,allChoices = prepForFitting_inferSerialHypoTesting(subject_data)
            # subject_data refers to posteriors and other matrices needed for model fitting, and choices is a dict storing the best-matching hypothesis for action in each trial
            results = likelihood_inferSerialHypoTesting(allChoices,subject_data,NChoices,NHyposAll)
            df = pd.DataFrame(results)

            all_data_df = pd.concat([all_data_df, df], ignore_index=True)
            all_data_df.to_csv('C:/Users/DELL/Desktop/humans-combine-value-learning-and-hypothesis-testing-main/SHT_llh_results.csv', index=False)
