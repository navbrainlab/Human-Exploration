import itertools
import random
import ast
import numpy as np
import pandas as pd
from env.game_env import Game
#Scarbu 2024.11
class Agent:
    def __init__(self, game_env):
        self.game_env = game_env
        self.food_dict=game_env.food_dict
        self.action_history = []
        self.reward_history = []
        self.block_weights = np.array([3, 2, 1], dtype=float)
    def choose_action(self ,episode, phase,round_num,choice_ls_0):
        block_1 = random.sample(choice_ls_0, 3)
        choice_ls_1 = [lst for lst in choice_ls_0 if lst not in block_1]
        block_2 = random.sample(choice_ls_1, 3)
        block_3 = [lst for lst in choice_ls_0 if lst not in block_2 + block_1]
        self.game_env.all_block_action = [block_1, block_2, block_3]
        # Save actions
        self.action_history.append({
            "episode": episode,
            "round": round_num,
            "action": self.game_env.all_block_action
        })
        return self.game_env.all_block_action
    def record_reward(self, episode, round_num, reward):
        '''Record reward for the current round'''
        self.reward_history.append({
            "episode": episode,
            "round": round_num,
            "reward": reward
        })

class Manual_Input_Agent:
    def __init__(self, game_env):
        self.game_env = game_env
        self.food_dict=game_env.food_dict
        self.action_history = []
        self.reward_history = []
        self.block_weights = np.array([3, 2, 1], dtype=float)  # 初始设定为 [3, 2, 1]

    def choose_action(self ,episode, phase,round_num,choice_ls_0):
        # block_1 = random.sample(choice_ls_0, 3)
        block_1_re=False
        while not block_1_re:
            try:
                user_input=input()
                block_1 = ast.literal_eval(user_input)
                block_1 = [[item.strip() for item in sublist] for sublist in block_1]
            except:
                print('input error')
                user_input = input()
                block_1 = ast.literal_eval(user_input)
                block_1 = [[item.strip() for item in sublist] for sublist in block_1]
            block_1_ls=[]
            try:
                for cob in block_1:
                    for idx,food in enumerate(cob):
                        # print(self.food_dict[phase][idx])
                        block_1_ls.append(self.food_dict[phase][idx].index(food)+1)
                block_1_re = True
            except:
                print('重新输入')
                block_1_re=False

        block_1_ls=np.reshape(block_1_ls,(-1,len(block_1[0]))).tolist()

        # block_1=[[3, 2, 1], [2, 3, 1], [1, 3, 3]]
        # choice_ls_1 = [lst for lst in choice_ls_0 if lst not in block_1]
        # block_2 = input()
        # block_2 = random.sample(choice_ls_1, 3)
        block_2_dup=False
        while not block_2_dup:
            try:
                user_input2=input()
                block_2=ast.literal_eval(user_input2)
                block_2 = [[item.strip() for item in sublist] for sublist in block_2]
            except:
                print('input error')
                user_input2 = input()
                block_2 = ast.literal_eval(user_input2)
                block_2 = [[item.strip() for item in sublist] for sublist in block_2]
            block_2_ls = []
            try:
                for cob in block_2:
                    for idx, food in enumerate(cob):
                        block_2_ls.append(self.food_dict[phase][idx].index(food) + 1)
                block_2_ls = np.reshape(block_2_ls, (-1, len(block_1[0]))).tolist()
                block_2_duplicate = [lst for lst in block_2_ls if lst in block_1_ls]
                if len(block_2_duplicate)>0:
                    print('第二行中不能出现第一行选过的内容,请重新选择第二行内容')
                    block_2_dup=False
                else:
                    block_2_dup = True
            except:
                print('重新输入')
                block_2_dup=False
        block_3 = [lst for lst in choice_ls_0 if lst not in block_1_ls + block_2_ls]


        self.game_env.all_block_action = [block_1_ls, block_2_ls, block_3]
        # Save actions
        self.action_history.append({
            "episode": episode,
            "round": round_num,
            "action": self.game_env.all_block_action
        })
        return self.game_env.all_block_action


    def record_reward(self, episode, round_num, reward):
        '''Record reward for the current round'''
        self.reward_history.append({
            "episode": episode,
            "round": round_num,
            "reward": reward
        })


import numpy as np
import random


class Qlearning_Agent:
    # [[[3, 2, 1], [1, 1, 1], [2, 1, 2]], [[2, 3, 1], [3, 1, 3], [1, 2, 2]], [[2, 2, 3], [1, 3, 3], [3, 3, 2]]]
    def __init__(self, game_env):
        self.game_env = game_env
        self.food_dict = game_env.food_dict
        self.action_history = []
        self.reward_history = []
        self.q_table = {}
        self.alpha = 0.2  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1  # 探索率
        self.current_state=0
        self.init=False
    def choose_action(self, episode, phase, round_num, choice_ls_0):
        self.current_state=(tuple(tuple(cob) for cob in choice_ls_0))
        # （Exploration）
        if not self.init:
            agent_action = self.explore_action(episode, choice_ls_0)
            self.init=True
        else:
            if random.uniform(0, 1) < self.epsilon:
                agent_action=self.explore_action(episode,choice_ls_0)

            # （Exploitation）
            else:
                agent_action=self.exploit_action(choice_ls_0)

        self.action_history.append(
                {"episode": episode, "round": round_num, "action": agent_action})

        return agent_action

    def explore_action(self, episode,choice_ls_0):
        # print('explore')
        #random choice
        # print(episode)
        random.seed(None)
        block_1 = random.sample(choice_ls_0, 3)
        choice_ls_1 = [lst for lst in choice_ls_0 if lst not in block_1]
        block_2 = random.sample(choice_ls_1, 3)
        block_3 = [lst for lst in choice_ls_0 if lst not in block_2 + block_1]
        block_1=(tuple(tuple(row) for row in block_1))
        block_2 = (tuple(tuple(row) for row in block_2))
        block_3 = (tuple(tuple(row) for row in block_3))
        return (block_1, block_2, block_3)

    def exploit_action(self, choice_ls_0):
        # print('exploit_action')
        max_q_value = -float('inf')
        best_action = None
        # for action in self.get_all_possible_actions(choice_ls_0):
            # action_tuple = tuple(tuple(row) for row in action)
            # q_value = self.get_q_value(action_tuple)
            # action = np.random.choice(np.where(q_value == q_value.max())[0])
        top_10_actions = sorted(self.q_table.items(), key=lambda item: item[1], reverse=True)[:5]
        best_action = random.choice(top_10_actions)[0][1]
        # print(best_action)


        return best_action

    def get_all_possible_actions(self, choice_ls_0):
        actions_ls=[]
        for first_row in itertools.combinations(choice_ls_0, 3):

            remaining = [item for item in choice_ls_0 if item not in first_row]

            for second_row in itertools.combinations(remaining, 3):
                third_row = [item for item in choice_ls_0 if item not in first_row+second_row]
                first_row = tuple(tuple(row) for row in first_row)
                second_row = tuple(tuple(row) for row in second_row)
                third_row = tuple(tuple(row) for row in third_row)
                actions_ls.append(tuple(map(tuple, (first_row,second_row,third_row))))

        return actions_ls

    def get_q_value(self, action):
        # print('get_q_value',action)
        state =self.current_state
        q_value = self.q_table.get((state, action), 0)
        return q_value

    def update_q_value(self, action, reward, next_state):
        # 更新Q值
        if reward==100:
            self.epsilon=0
        state = self.current_state
        if next_state is None:
            self.q_table[(state, tuple(action))] = self.q_table.get((state, tuple(action)), 0) + self.alpha * (
                        reward - self.get_q_value(action))
        else:
            best_next_action = max(self.get_all_possible_actions(next_state), key=lambda x: self.get_q_value(x))
            self.q_table[(state, tuple(action))] = self.q_table.get((state, tuple(action)), 0) + self.alpha * (
                        reward + self.gamma * self.get_q_value(best_next_action) - self.get_q_value(action))


    def convert_action_to_tuple(self, action):

        return tuple(tuple(block) for block in action)
    def record_reward(self, episode, round_num, reward,next_state):
        '''Record reward for the current round'''
        self.reward_history.append({
            "episode": episode,
            "round": round_num,
            "reward": reward
        })


        # next_state = self.get_state_from_action(self.game_env.next_round())
        self.update_q_value( tuple(self.game_env.agent_action), reward, next_state)



