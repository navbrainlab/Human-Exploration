import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import os
# agent_dict = {'agent_id':[],'round':[],'score':[],
#               'choice_1':[],'choice_2':[],'choice_3':[],'choice_4':[],
#               'pos_1':[],'pos_2':[],'pos_3':[],'pos_4':[],}
def choice_partten_locate(choice_reward_pair,gap_range,choice_pattern,choice_seq):

    score = choice_reward_pair[choice_reward_pair['choice_seq'].apply(lambda x: np.array_equal(x, choice_seq))]['value'].tolist()[0]
    # score = choice_reward_pair[choice_reward_pair['']==choice_seq]['value'].tolist()[0]
    if choice_pattern == '122' :
        gap = gap_range.loc[gap_range['choices_category']==122]['gap'].tolist()[0] # -10.741
        choice_locate = score + gap
    elif choice_pattern == '222':
        gap = gap_range.loc[gap_range['choices_category']==222]['gap'].tolist()[0] #89.259
        choice_locate = score + gap
    elif choice_pattern == '111' :
        gap = gap_range.loc[gap_range['choices_category']==111]['gap'].tolist()[0] #189.259
        choice_locate = score + gap
    elif choice_pattern == '223':
        gap = gap_range.loc[gap_range['choices_category']==223]['gap'].tolist()[0] #289.259
        choice_locate = score +gap
    elif choice_pattern == '333':
        gap = gap_range.loc[gap_range['choices_category']==333]['gap'].tolist()[0]#389.259
        choice_locate = score +gap
    return choice_locate

# agent_dict = pd.read_csv(r'C:\Users\Windows11\Desktop\agent1\agent_dict.csv')
def choice_reward_pic(save_path,nD,agent_dict):
    
    pic_path = r'results/naive_RL'+'/'+save_path
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    choice_reward_pair=pd.read_csv('choice_reward_pair.csv')
    # choice_reward_pair=pd.read_csv(r'C:\Users\Windows11\Desktop\choice_reward_pair.csv')
    choice_reward_pair['choice_seq'] = choice_reward_pair['blocks'].apply(lambda x: np.array(ast.literal_eval(x)))
    
    range_minmax=choice_reward_pair.groupby('choices_category')['X_coord'].agg(['min','max']).reset_index()

    choice_reward_pair['gap'] = choice_reward_pair['X_coord']-choice_reward_pair['value']
    gap_range=choice_reward_pair.groupby('choices_category')['gap'].agg('min').reset_index()
    

    for j in range(nD):
        pattern_list = 'pattern_'+str(j+1)
        choice_list = 'choice_'+str(j+1)
        pos_list = 'pos_'+str(j+1)

        for i in range(len(agent_dict[choice_list])):
            choice_locate = choice_partten_locate(choice_reward_pair,gap_range,agent_dict[pattern_list][i],agent_dict[choice_list][i])
            agent_dict[pos_list].append(choice_locate)
    # print([len(v) for v in agent_dict.values()])
    # print(agent_dict.keys())

    df_data = pd.DataFrame(agent_dict)
    
    df_data.to_csv(pic_path+'/'+'agent_dict_'+str(df_data['agent_id'].unique()[0])+'.csv')
    for subject in df_data['agent_id'].unique():
        df_sub=df_data[df_data.agent_id==subject]

        df =df_sub.reset_index(drop=True)

        

        color_ls=['blue', 'green', 'darkorange','purple']
        plt.figure(figsize=(21,15))
        for i,dim in enumerate(['pos_1','pos_2','pos_3']):
        #     plt.figure()
            color_dict={'333':'lightblue','223':'lightcoral','222':'lightyellow','122':'lightgreen','111':'purple','Points':'gray','angle_between_vec':'orange'}
            for category, group in range_minmax.groupby('choices_category')[['min','max']]:
                plt.axvspan(group['min'].values[0],group['max'].values[0], color=color_dict[str(category)], alpha=0.2)

            plt.title(f'Subject {subject} Dim{i+1} visualization')
            plt.plot(df[dim],df['round'],marker='o',label=f'dim{i+1}',c=color_ls[i],markersize=5)
        plt.axvspan(500-27.593,600-27.593, color=color_dict['Points'], alpha=0.2)
        plt.axvline(550-27.593, color='red',linestyle='dashdot', alpha=0.2)


        plt.plot(df['score']+550-50-27.593,df['round'],marker='o',label=f'points',c='r',markersize=5)


        plt.xticks([0,50-27.593,46,89,150-27.593 ,250-27.593,350-27.593,389,450-27.593,500-27.593,550-27.593,600-27.593],
                ['0','122','46','89','222','111','223','389','333','0','Points','100'],fontsize=15)



        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        # pic_path = r'C:\Users\Windows11\Desktop\fRL_agent100'
        plt.savefig(pic_path +'/'+str(subject)+'.png')

# choice_reward_pic(4,agent_dict)