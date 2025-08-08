import numpy as np
import pandas as pd

from agent import Manual_Input_Agent,Agent,Qlearning_Agent
from env.game_env import Game
#Scarbu 2024.11



def train_rl_agent(num_episodes, phase='P1',show_word=True,save_to_csv=False):
    game_env = Game()
    agent = Qlearning_Agent(game_env)
    episode_rewards = []
    first_state=[[3, 3, 2], [3, 1, 3], [2, 3, 1], [2, 1, 2], [2, 2, 3], [1, 1, 1], [1, 3, 3], [1, 2, 2], [3, 2, 1]]
    for episode in range(num_episodes):
        # if episode % 50 == 0:
        #     print('Episode {} of {}'.format(episode, num_episodes))
        print(f"Starting episode {episode}")
        game_env.reset()
        game_env.load_round_env(phase)
        total_reward = 0
        state=game_env._get_state()


        while state:
            if show_word:
                state_word_ls = []
                for cob in state:
                    for idx in range(len(cob)):
                        num2word=game_env.food_dict[phase][idx][cob[idx]-1]
                        state_word_ls.append(num2word)

                state_word_ls=np.reshape(state_word_ls, (-1, len(state[0])))
                print(f'第{game_env.round_num}轮的选项为：',state_word_ls.tolist(),'请说明你将如何将他们分布在三行中请不要使用英文,完成选择三行选择后再陈述你的思考过程')
            round_num = game_env.round_num
            actions = agent.choose_action(episode, phase,round_num,state)
            # print('agent actions',actions)
            state, reward=game_env.step(actions)

            # print(reward)
            # print(f'你第{round_num}轮的得分为：', reward,f',剩余{game_env.total_round-round_num}轮')
            # reward = game_env.calc_reward(actions)
            # agent.update_q_value(state.)
            # print(state)

            if reward!=100:
                reward=0.1
            agent.record_reward(episode, round_num, reward,state)
            total_reward += reward
            game_env.load_round_env(phase)

        episode_rewards.append(total_reward)
        df=pd.DataFrame(agent.reward_history)
        # if round(df[df['episode']==episode]['reward'].max())==100:
        print(f"Episode {episode} completed with mean reward: {round(df[df['episode']==episode]['reward'].median())},"
              f"max reward: {round(df[df['episode']==episode]['reward'].max())} ")


    # Save history to files
    if save_to_csv:
        pd.DataFrame(agent.action_history).to_csv(f"actions_history{game_env.reward_dim}.csv", index=False)
        pd.DataFrame(agent.reward_history).to_csv(f"rewards_history{game_env.reward_dim}.csv", index=False)


    df = pd.DataFrame.from_dict(agent.q_table, orient='index')
    df.to_csv('q_table.csv')
    print("Training completed.")
    return episode_rewards

if __name__ == "__main__":
    train_rl_agent(num_episodes=100, phase='P1',save_to_csv=False)