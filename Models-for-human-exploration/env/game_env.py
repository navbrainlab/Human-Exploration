import os
import random
import numpy as np
import pandas as pd
#Scarbu 2024.11
## chice part
class Env(object):
    def __init__(self):
        '''Initialize a game'''
        self.init=False
        self.total_round=0
        self.dim_df=pd.DataFrame()
        self.round_num=0
        self.all_comb = []
        self.reward=0
        self.food_dict={
        'P1':{
            0:['棒棒糖','饼干','小蛋糕'],
            1:['披萨','热狗','汉堡'],
            2:['果汁','果酒','牛奶']},
        'P2':{
            0:['甜筒','蛋糕','甜甜圈'],
            1:['米饭','面条','寿司'],
            2:['葡萄','香蕉','苹果'],
            3:['啤酒','红酒','咖啡']
        }}

    def reset(self):
        '''Reset the environment'''
        # self.reward_dim = np.sort(random.sample([0, 1, 2], 2))
        self.init = False
        self.total_round = 0
        self.dim_df = pd.DataFrame()
        self.round_num = 0
        self.all_comb = []
        self.reward = 0
        self.agent_action=[]
        return self._get_state()

    def load_round_env(self, phase, file_path):
        '''Load the environment'''
        if not self.init:
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError("Environment configuration file not found.")
            if phase == 'P1':
                self.reward_dim = np.sort(random.sample([0,1,2] ,2))
            else:
                self.reward_dim = np.sort(random.sample([0, 1, 2, 3], 2))
            self.reward_dim=[1,2]
            self.dim_df = pd.read_csv(file_path)
            self.total_round = 30 if phase == 'P1' else 50
            self.init = True
            # print('Main dim:', self.reward_dim)

        if self.round_num>self.total_round-1:
            return None
        random.seed(self.round_num)
        column_values=self.dim_df[str(self.round_num)].values
        shuffled_values = column_values.copy()
        random.shuffle(shuffled_values)
        self.all_comb=[eval(i) for i in shuffled_values]
        return self.dim_df

    def next_round(self):
        self.round_num += 1
        random.seed(self.round_num)
        if self.round_num >self.total_round-1:
            return None
        if self.reward == 100:
            return None
        column_values = self.dim_df[str(self.round_num)].values
        shuffled_values = column_values.copy()
        random.shuffle(shuffled_values)
        self.all_comb = [eval(i) for i in shuffled_values]
        return self.all_comb

    def step(self, all_block_action):

        self.agent_action=all_block_action
        '''Take an action and return the next state, reward, done flag, and info'''
        reward = self.calc_reward(all_block_action)
        next_state = self.next_round()
        return next_state, reward

    def _get_state(self):
        '''Return current state that the agent can observe,
        what constitutes the state of the game that is visible to the agent.'''

        return self.all_comb
    def calc_reward(self,all_block_action):
        '''calculate the reward'''
        level_order_reward={'1':10,'2':5,'3':1}
        block_reward=[]
        #print(f'The important dimensions are {self.reward_dim[0],self.reward_dim[1]}')
        for i in all_block_action:
            Main_dim_1_reward = 0
            Main_dim_2_reward = 0
            
            #print(i)
            Main_dim_1_reward+=level_order_reward[str(i[self.reward_dim[0]])]
            Main_dim_2_reward += level_order_reward[str(i[self.reward_dim[1]])]
            #print()
            block_reward.append(Main_dim_1_reward+Main_dim_2_reward)
            #print(Main_dim_1_reward+Main_dim_2_reward)    # reward of every line
        block_reward = np.array(block_reward)
        #print(block_reward)
        block_reward = block_reward.reshape(3,3)
        #print(block_reward)
        weighted_block_reward=(np.array(block_reward).T*np.array([10,5,1])).T
        #print(weighted_block_reward)
        #print(sum(weighted_block_reward))
        all_reward=(sum(sum(weighted_block_reward))-350)*90/324+10
        if all_reward not in [0, 100]:
            noise=np.random.uniform(-2,2)
            all_reward+=noise
        self.reward=round(all_reward)
        return self.reward

# if __name__=='__main__':
#     print(game().load_round_env('P1'))




