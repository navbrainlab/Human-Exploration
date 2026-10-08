import numpy as np
import pandas as pd
import os
import random
import ast
# import pic_draw
from scipy.special import logsumexp
from itertools import combinations
from taskSetting_jilab import hypotheses_3D as hypotheses
from taskSetting_jilab import cost_for_h

# =============================================================================

def define_choice_pattern(phase, agent_dict, act_data):
    nD = 3 if phase == 'p1' else 4;
    act_data = np.array(act_data)
    for i in range(nD):
        dim_list = 'pattern_' + str(i + 1);
        choice_seq_list = 'choice_' + str(i + 1)
        dim_choice = act_data[:, :, i].reshape(--1, 3);
        sorted_arr = np.sort(dim_choice, axis=1)
        dim_choice_flat = sorted_arr.flatten().tolist();
        choice_pattern = []
        for food in range(3):
            max_food_count = 0
            for j in range(3):
                row_count = len([x for x in dim_choice_flat[j * 3:j * 3 + 3] if x == food + 1])
                if row_count > max_food_count: max_food_count = row_count
            choice_pattern.append(max_food_count)
        all_row_count = ''.join(str(i) for i in sorted(choice_pattern))
        agent_dict[dim_list].append(all_row_count);
        agent_dict[choice_seq_list].append(dim_choice_flat)
    return agent_dict


def get_options(trial, phase, options_3D, options_4D):
    options_str = options_3D.iloc[:, trial].tolist() if phase == "p1" else options_4D.iloc[:, trial].tolist()
    return [ast.literal_eval(x) for x in options_str]


def hypothesisToAction(h_matrix, options):
    def score_func(option, h): return np.nansum([h[i][int(option[i]) - 1] for i in range(len(option))])

    shuffled_options = options[:];
    random.shuffle(shuffled_options)
    options_score_sort = sorted(shuffled_options, key=lambda opt: score_func(opt, h_matrix), reverse=True)
    return [options_score_sort[i:i + 3] for i in range(0, len(options_score_sort), 3)]


def rewardFunc(action):
    level_order_reward = {'1': 10, '2': 5, '3': 1};
    block_reward = [];
    reward_dim = [0, 1]
    for i in action:
        for j in i:
            r1 = level_order_reward[str(j[reward_dim[0]])];
            r2 = level_order_reward[str(j[reward_dim[1]])]
            block_reward.append(r1 + r2)
    block_reward_grid = np.array(block_reward).reshape(3, 3)
    weighted_block_reward = (block_reward_grid.T * np.array([10, 5, 1])).T
    all_reward = (sum(sum(weighted_block_reward)) - 350) * 90 / 324 + 10
    if all_reward not in [0, 100]: all_reward += np.random.uniform(-2, 2)
    return round(all_reward)


def hypothesisToFeatureMat(hypotheses_list, numDimensions, numFeaturesPerDimension):
    featureMat = np.zeros([len(hypotheses_list), numDimensions * numFeaturesPerDimension])
    for i_h, h in enumerate(hypotheses_list):
        for i_dim in range(numDimensions):
            active_feature_idx = next((i for i, x in enumerate(h[i_dim]) if not np.isnan(x)), -1)
            if active_feature_idx != -1:
                mat_idx = i_dim * numFeaturesPerDimension + active_feature_idx
                featureMat[i_h, mat_idx] = 1
    return featureMat


def stimulusToFeatureMat(action_grid, numDimensions, numFeaturesPerDimension):
    featureMat = np.zeros(numDimensions * numFeaturesPerDimension);
    row_weight = [10, 5, 1]
    # row_weight = [0.5, 0, -0.5]

    food_score = {k: [0, 0, 0] for k in range(numDimensions)}
    for row_idx, row in enumerate(action_grid):
        for item in row:
            for dim_idx, feature_val in enumerate(item):
                food_score[dim_idx][int(feature_val - 1)] += row_weight[row_idx]
    for dim_idx, scores in food_score.items():
        for feature_idx, score in enumerate(scores):
            mat_idx = dim_idx * numFeaturesPerDimension + feature_idx;
            featureMat[mat_idx] = score
    norm = np.linalg.norm(featureMat)
    norm=1
    if norm > 0: featureMat = featureMat / norm
    return featureMat

def q_feature_decay(Q_features,feature_vec_stimulus):
    d_vec=np.array([0.5,0.5,0.6,0.9,0.95,1,1])
    decay_idx = ((feature_vec_stimulus + 1.5) / 0.5).astype(int)
    decay_vec = d_vec[decay_idx]

    Q_features = decay_vec * Q_features

    return Q_features

def Credict_assignmeant(Q_features,reward,feature_vec_stimulus):
    contribute=feature_vec_stimulus/144
    credict=contribute*(2*reward-100)
    # credict =contribute*reward
    rpe=credict/ np.sum(abs(credict))-Q_features
    Q_features+=eta*(rpe)
    return Q_features

def sigmoid(x):
    """标准sigmoid函数，将值映射到(0,1)区间"""
    return 1 / (1 + np.exp(-x-1.5))
def softmax(vector):
    # 计算向量中每个元素的指数
    exp_vector = np.exp(vector)

    # 计算指数向量的和
    sum_exp = np.sum(exp_vector)

    # 计算softmax结果
    softmax_result = exp_vector / sum_exp

    return softmax_result
def YY_fRL(numDimensions,input_data,W,reward,eta,decay=False):
    W=np.array(W).reshape(numDimensions,3)
    input_data=np.array(np.concatenate(input_data).tolist())
    Q_values = []
    coefficient = [0.5, 0, -0.5]
    decay_matrix=[1,0.8,0.5]
    for i in range(input_data.shape[0]):
        q_val = 0
        for j in range(input_data.shape[1]):
            q_val += (W[j][input_data[i][j] - 1])
        Q_values.append(q_val)

    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            # rpe = (2*reward-100) / 165 - Q_values[i]
            rpe =  reward  / 100 - Q_values[i]
            if decay:
                W[j][input_data[i][j] - 1] += coefficient[int(i / 3)] * eta * rpe*decay_matrix[int(i / 3)]

            else:
                W[j][input_data[i][j] - 1] += coefficient[int(i / 3)] * eta * rpe
            # W[j][input_data[i][j] - 1] += coefficient[int(i / 3)] * eta * rpe
            # decay


    return  W.reshape(numDimensions*3,1).flatten().tolist()
def minmax_norm_simple(arr):
    arr=np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    # 处理所有元素相同的特殊情况，避免除零
    return (arr - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(arr)

# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == '__main__':
    # --- Game and Model Setup ---
    phase = 'p1';
    nF = 3;
    nD = 3;
    decay=True
    numDimensions = nD;
    numFeaturesPerDimension = nF;
    gameLength = 30
    NHypos = len(hypotheses)
    options_3D = pd.read_csv("data/options_3_Dimension.csv")
    options_4D = pd.read_csv("data/options_4_Dimension.csv")
    featureMatAllHypotheses = hypothesisToFeatureMat(hypotheses, numDimensions, numFeaturesPerDimension)

    # --- Start Agent Simulation Loop ---
    for agent in range(100):
        print(f"--- Running Agent {agent} ---")

        # --- Model Parameters ---
        # betaStay = 8
        # # [MODIFIED] thetaStay is now a probability threshold between 0 and 1
        # thetaStay = 0.25  # A neutral satisfaction threshold
        #
        # betaSwitch = 3
        # eta = 0.3
        # epsilon = 0.02
        betaStay = 10
        # [MODIFIED] thetaStay is now a probability threshold between 0 and 1
        thetaStay = 0.5  # A neutral satisfaction threshold
        thetaTest=0.5
        betaSwitch = 3
        eta = 0.3
        epsilon = 0.02

        # --- Data Logging Setup ---

        if decay:
            ori_save_pth= 'serial_hypothesis_song_model_decay'
        else:
            ori_save_pth = 'serial_hypothesis_song_model'
        save_pth = f'./{ori_save_pth}/{agent}_fitting.csv'
        os.makedirs(os.path.dirname(save_pth), exist_ok=True)
        # [MODIFIED] Log counters instead of LR
        data_columns = ['agent', 'trial', 'reward', 'h_idx', 'action', 'p_stay',
                        'rewards_for_h', 'trials_for_h', 'q_features']
        pd.DataFrame(columns=data_columns).to_csv(save_pth, index=False)

        # --- [MODIFIED] Agent State Initialization ---
        # current_hypothesis_idx = np.random.randint(NHypos)
        current_hypothesis_idx = 2
        # Reverted back to counters
        trials_for_current_h = 0
        rewards_for_current_h = 0

        Q_features = np.ones(numDimensions * numFeaturesPerDimension)*1/(numDimensions * numFeaturesPerDimension)

        # ====================================================================
        # TRIAL LOOP START
        # ====================================================================
        for trial in range(gameLength):
            # --- 1. DECIDE: Stay with the current hypothesis or Switch? ---
            # [MODIFIED] p_stay is now calculated using reward probability counting
            if trials_for_current_h > 0:
                # The (rewards + 1) / (trials + 2) is a Bayesian update for a probability (Beta-Binomial)
                # This prevents division by zero and acts as a regularizing prior. It's good practice.
                estimated_reward_prob = (rewards_for_current_h + 1) / (trials_for_current_h + 2)
            else:
                # If there's no evidence yet, the prior probability is 0.5
                estimated_reward_prob = 1

            p_stay = 1 / (1 + np.exp(-betaStay * (estimated_reward_prob - thetaStay)))

            if np.random.random() < p_stay:
                pass  # STAY
            else:  # SWITCH
                expected_R_hypotheses = np.dot(featureMatAllHypotheses, Q_features)
                                         # -cost_for_h)
                if max(expected_R_hypotheses) < thetaTest:
                    current_hypothesis_idx=np.random.choice(NHypos)
                else:
                    log_p_new_h = betaSwitch * expected_R_hypotheses - logsumexp(betaSwitch * expected_R_hypotheses)
                    p_new_h = np.exp(log_p_new_h+ np.log(1-p_stay))
                    # new_hypothesis_idx = np.random.choice(NHypos, p=p_new_h)
                    new_hypothesis_idx=np.argmax(p_new_h)
                    # Update agent's state to reflect the switch
                    current_hypothesis_idx = new_hypothesis_idx
                    # Reset counters for the new hypothesis
                trials_for_current_h = 0
                rewards_for_current_h = 0

            # --- 2. ACT: Make a choice (Unchanged) ---
            current_hypothesis_matrix = hypotheses[current_hypothesis_idx]
            options = get_options(trial, phase, options_3D, options_4D)
            if np.random.random() < epsilon:
                random_h_for_action = hypotheses[np.random.randint(NHypos)]
                action = hypothesisToAction(random_h_for_action, options)
            else:
                action = hypothesisToAction(current_hypothesis_matrix, options)

            # --- 3. GET REWARD & UPDATE STATES ---
            reward = rewardFunc(action)
            if trial==0:
                print(reward)
            is_rewarded = reward > 70

            # --- [MODIFIED] Update SHT evidence using simple counting ---
            trials_for_current_h += 1
            if is_rewarded:
                rewards_for_current_h += 1

            # --- Update the Q_features by fRL  ---
            feature_vec_stimulus = stimulusToFeatureMat(action, numDimensions, numFeaturesPerDimension)
            # ============ our fRL==================
            Q_features=YY_fRL(numDimensions,action,Q_features,reward,eta,decay)
            Q_features=softmax(Q_features)
            # Q_features=Credict_assignmeant(Q_features,reward,feature_vec_stimulus)
            # ============ our fRL==================

            # prediction = np.dot(feature_vec_stimulus, Q_features)
            # outcome = reward / 165
            # rpe = outcome-prediction
            # Q_features += eta * rpe * feature_vec_stimulus



            # --- 4. DATA LOGGING ---
            row = {
                "agent": agent, "trial": trial, "reward": reward,
                "h_idx": current_hypothesis_idx, "action": str(action),
                "p_stay": p_stay, "rewards_for_h": rewards_for_current_h,
                "trials_for_h": trials_for_current_h,
                "q_features": str(list(np.round(Q_features, 3)))
            }
            pd.DataFrame([row]).to_csv(save_pth, mode="a", header=False, index=False)

            if reward == 100:
                print(f"Agent {agent} found the solution on trial {trial}.")
                break

    print("All agent simulations finished.")