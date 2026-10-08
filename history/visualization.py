import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# choice_ls=[
#     [[1, 1, 1], [2, 1, 2], [3, 2, 1], [3, 1, 3], [1, 2, 2], [2, 3, 1], [2, 2, 3], [3, 3, 2], [1, 3, 3]],
# [[2, 1, 2], [1, 2, 2], [3, 3, 2], [3, 1, 1], [3, 2, 3], [1, 1, 3], [2, 3, 3], [1, 3, 1], [2, 2, 1]],
# [[3, 1, 3], [2, 2, 3], [1, 3, 3], [3, 3, 1], [1, 1, 2], [3, 2, 2], [1, 2, 1], [2, 3, 2], [2, 1, 1]],
# [[3, 3, 3], [1, 2, 3], [2, 1, 3], [3, 1, 1], [2, 2, 1], [1, 3, 1], [1, 1, 2], [3, 2, 2], [2, 3, 2]],
# [[2, 2, 2], [3, 1, 2], [1, 3, 2], [3, 2, 3], [1, 1, 3], [2, 3, 3], [3, 3, 1], [1, 2, 1], [2, 1, 1]],
# [[1, 1, 1], [3, 2, 1], [2, 3, 1], [1, 3, 2], [3, 1, 2], [2, 2, 2], [3, 3, 3], [1, 2, 3], [2, 1, 3]],
# [[1, 1, 1], [1, 2, 2], [3, 1, 2], [2, 1, 3], [1, 3, 3], [3, 2, 3], [2, 2, 1], [3, 3, 1], [2, 3, 2]],
# [[3, 2, 1], [1, 2, 2], [3, 1, 2], [3, 3, 3], [2, 2, 3], [1, 1, 3], [1, 3, 1], [2, 1, 1], [2, 3, 2]],
# [[2, 3, 1], [3, 3, 2], [2, 1, 3], [1, 3, 3], [2, 2, 2], [3, 2, 3], [3, 1, 1], [1, 2, 1], [1, 1, 2]],
# [[3, 2, 1], [1, 2, 3], [3, 1, 3], [3, 3, 2], [1, 1, 2], [1, 3, 1], [2, 2, 2], [2, 3, 3], [2, 1, 1]],
# [[2, 1, 2], [2, 3, 1], [3, 3, 3], [1, 3, 2], [2, 2, 3], [1, 2, 1], [1, 1, 3], [3, 1, 1], [3, 2, 2]],
# [[1, 1, 1], [2, 1, 2], [3, 2, 2], [1, 2, 3], [3, 1, 3], [2, 2, 1], [1, 3, 2], [2, 3, 3], [3, 3, 1]],
# [[1, 1, 1], [3, 2, 1], [1, 2, 2], [2, 1, 2], [3, 1, 3], [2, 3, 1], [3, 3, 2], [1, 3, 3], [2, 2, 3]],
# [[1, 2, 2], [2, 1, 2], [2, 2, 1], [3, 3, 2], [2, 3, 3], [3, 1, 1], [1, 3, 1], [1, 1, 3], [3, 2, 3]],
# [[3, 1, 3], [1, 3, 3], [2, 2, 3], [1, 2, 1], [3, 3, 1], [1, 1, 2], [2, 3, 2], [2, 1, 1], [3, 2, 2]],
# [[2, 2, 1], [1, 2, 3], [1, 1, 2], [2, 1, 3], [3, 3, 3], [1, 3, 1], [3, 2, 2], [3, 1, 1], [2, 3, 2]],
# [[3, 1, 2], [1, 2, 1], [1, 3, 2], [2, 3, 3], [2, 1, 1], [3, 2, 3], [1, 1, 3], [2, 2, 2], [3, 3, 1]],
# [[1, 1, 1], [3, 2, 1], [3, 1, 2], [2, 3, 1], [1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 3, 3], [2, 2, 2]],
# [[1, 1, 1], [1, 2, 2], [3, 1, 2], [2, 2, 1], [2, 1, 3], [2, 3, 2], [1, 3, 3], [3, 2, 3], [3, 3, 1]],
# [[3, 2, 1], [1, 2, 2], [3, 1, 2], [3, 3, 3], [2, 1, 1], [2, 3, 2], [2, 2, 3], [1, 3, 1], [1, 1, 3]],
# [[2, 3, 1], [3, 3, 2], [2, 1, 3], [1, 2, 1], [1, 1, 2], [1, 3, 3], [2, 2, 2], [3, 1, 1], [3, 2, 3]],
# [[3, 2, 1], [1, 2, 3], [3, 1, 3], [1, 1, 2], [3, 3, 2], [2, 3, 3], [2, 1, 1], [1, 3, 1], [2, 2, 2]],
# [[2, 1, 2], [1, 3, 2], [2, 3, 1], [1, 2, 1], [3, 3, 3], [3, 1, 1], [2, 2, 3], [3, 2, 2], [1, 1, 3]],
# [[1, 1, 1], [2, 1, 2], [2, 2, 1], [1, 3, 2], [1, 2, 3], [3, 1, 3], [3, 2, 2], [2, 3, 3], [3, 3, 1]],
# [[1, 1, 1], [1, 2, 2], [3, 2, 1], [2, 1, 2], [2, 3, 1], [3, 1, 3], [3, 3, 2], [2, 2, 3], [1, 3, 3]],
# [[1, 2, 2], [2, 1, 2], [2, 2, 1], [3, 1, 1], [2, 3, 3], [3, 3, 2], [1, 3, 1], [1, 1, 3], [3, 2, 3]],
# [[3, 1, 3], [1, 1, 2], [1, 2, 1], [2, 1, 1], [3, 3, 1], [3, 2, 2], [1, 3, 3], [2, 2, 3], [2, 3, 2]],
# [[2, 2, 1], [1, 1, 2], [1, 2, 3], [3, 1, 1], [3, 2, 2], [2, 3, 2], [1, 3, 1], [2, 1, 3], [3, 3, 3]],
# [[3, 1, 2], [1, 2, 1], [1, 3, 2], [2, 1, 1], [2, 2, 2], [1, 1, 3], [3, 2, 3], [2, 3, 3], [3, 3, 1]],
# [[1, 1, 1], [3, 2, 1], [2, 2, 2], [3, 3, 3], [2, 1, 3], [3, 1, 2], [1, 3, 2], [2, 3, 1], [1, 2, 3]],
# [[1, 1, 1], [1, 2, 2], [2, 2, 3], [1, 3, 3], [3, 3, 2], [2, 1, 2], [3, 2, 1], [2, 3, 1], [3, 1, 3]],

# ]
# score_ls=np.array([54, 56, 53, 90, 39, 36, 89, 60, 69, 54, 70, 49, 63, 67, 50, 69, 50, 60, 67, 53, 67, 66, 78, 87, 87, 77, 81, 68, 75, 79, 88])
class visualization_r_a:
    def __init__(self):
        self.choice_ls = []
        self.score_ls = []

    def update(self, choice_ls, score_ls):
        self.choice_ls = choice_ls
        self.score_ls = score_ls

    def vis(self):
        choice_ls=np.reshape(self.choice_ls,(len(self.choice_ls),3,3,np.shape(self.choice_ls)[-1]))

        D=np.shape(choice_ls)[-1]
        data = []
        P1P2={}
        idx=0

        for choice in choice_ls:
            
            one_round=choice
            init=False
            block_list=[None]*4
            for blocks in one_round:
                blocks_matrix=[]
                for ob in blocks:
                    blocks_matrix.append((ob))
                blocks_matrix_arr = np.array(blocks_matrix)
                if not init:
                    for i in range(blocks_matrix_arr.shape[1]):
                        block_list[i]=list(np.sort(blocks_matrix_arr[:,i]))
                        init=True
                else:
                    for i in range(blocks_matrix_arr.shape[1]):
                        block_list[i].extend( np.sort(list(blocks_matrix_arr[:,i])))


            if D == 3:
                data.extend([{'round':idx+1,
                            'block1':choice_ls[:3],'block2':choice_ls[3:6],
                            'block3':choice_ls[6:9],'dim1':tuple(block_list[0]),'dim2':tuple(block_list[1]),'dim3':tuple(block_list[2])
                            }])
            else:
                data.extend([{'round':idx+1,
                                'block1':choice_ls[:3],'block2':choice_ls[3:6],
                                'block3':choice_ls[6:9],'dim1':tuple(block_list[0]),'dim2':tuple(block_list[1]),'dim3':tuple(block_list[2]),'dim4':tuple(block_list[3])
                                    }])
            idx+=1
        df_data = pd.DataFrame(data)
        df=pd.read_csv(r'choice_reward_pair.csv')
        df_data.loc[:, 'X']=df_data['dim1'].apply(lambda row: df[df['blocks'].isin([str(row)])]['X_coord'].values[0])
        df_data.loc[:, 'dim1_choice_value']=df_data['dim1'].apply(lambda row: df[df['blocks'].isin([str(row)])]['value'].values[0])
        df_data.loc[:, 'dim1_category']=df_data['dim1'].apply(lambda row: df[df['blocks'].isin([str(row)])]['choices_category'].values[0])

        df_data.loc[:, 'Y']=df_data['dim2'].apply(lambda row: df[df['blocks'].isin([str(row)])]['X_coord'].values[0])
        df_data.loc[:, 'dim2_choice_value']=df_data['dim2'].apply(lambda row: df[df['blocks'].isin([str(row)])]['value'].values[0])
        df_data.loc[:, 'dim2_category']=df_data['dim2'].apply(lambda row: df[df['blocks'].isin([str(row)])]['choices_category'].values[0])

        df_data.loc[:, 'Z']=df_data['dim3'].apply(lambda row: df[df['blocks'].isin([str(row)])]['X_coord'].values[0])
        df_data.loc[:, 'dim3_choice_value']=df_data['dim3'].apply(lambda row: df[df['blocks'].isin([str(row)])]['value'].values[0])
        df_data.loc[:, 'dim3_category']=df_data['dim3'].apply(lambda row: df[df['blocks'].isin([str(row)])]['choices_category'].values[0])
        if D != 3:
            df_data.loc[:, 'T']=df_data['dim4'].apply(lambda row: df[df['blocks'].isin([str(row)])]['X_coord'].values[0] if isinstance(row, (list, tuple)) else None)
            df_data.loc[:, 'dim4_choice_value']=df_data['dim4'].apply(lambda row: df[df['blocks'].isin([str(row)])]['value'].values[0] if isinstance(row, (list, tuple)) else None)
            df_data.loc[:, 'dim4_category']=df_data['dim4'].apply(lambda row: df[df['blocks'].isin([str(row)])]['choices_category'].values[0] if isinstance(row, (list, tuple)) else None)
        choice_reward_pair=pd.read_csv(r'choice_reward_pair.csv')
        range_minmax=choice_reward_pair.groupby('choices_category')['X_coord'].agg(['min','max']).reset_index()

        color_ls=['blue', 'green', 'darkorange','purple']
        plt.figure(figsize=(12,8))
        if D != 3:
            for i,dim in enumerate(['X','Y','Z','T']):
            #     plt.figure()
                color_dict={'333':'lightblue','223':'lightcoral','222':'lightyellow','122':'lightgreen','111':'purple','Points':'gray','angle_between_vec':'orange'}
                for category, group in range_minmax.groupby('choices_category')[['min','max']]:
                    plt.axvspan(group['min'].values[0],group['max'].values[0], color=color_dict[str(category)], alpha=0.2)
            
                plt.title(f'RL Dim{i+1} visualization')
                
                plt.plot(df_data[dim],df_data['round'],marker='o',label=f'dim{i+1}',c=color_ls[i],markersize=5)
        else:
            for i,dim in enumerate(['X','Y','Z']):
            #     plt.figure()
                color_dict={'333':'lightblue','223':'lightcoral','222':'lightyellow','122':'lightgreen','111':'purple','Points':'gray','angle_between_vec':'orange'}
                for category, group in range_minmax.groupby('choices_category')[['min','max']]:
                    plt.axvspan(group['min'].values[0],group['max'].values[0], color=color_dict[str(category)], alpha=0.2)
            
                plt.title(f'RL Dim{i+1} visualization')
                
                plt.plot(df_data[dim],df_data['round'],marker='o',label=f'dim{i+1}',c=color_ls[i],markersize=5)
        plt.axvspan(500-27.593,600-27.593, color=color_dict['Points'], alpha=0.2)
        plt.axvline(550-27.593, color='red',linestyle='dashdot', alpha=0.2)

        plt.axvspan(650-27.593,740-27.593, color=color_dict['angle_between_vec'], alpha=0.2)
        plt.plot(self.score_ls+550-50-27.593,df_data['round'],marker='o',label=f'points',c='r',linestyle=':',markersize=5)
        plt.xticks([50-27.593,150-27.593,250-27.593,350-27.593,450-27.593,500-27.593,550-27.593,600-27.593,650-27.593,740-27.593],
                ['122','222','111','223','333','0','Points','100','0','90'],fontsize=15)

        plt.ylabel('Round number',fontsize=15)
        plt.xlabel('Choices category',fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(axis='y')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        # plt.show()
        plt.savefig(f'reward_action.png')  
        plt.close()
        print(f'Picture saved')