clear;
clc;
rng(44501)
dim = 4;
% load('../Mydata_relirr_nfb.mat','Mydata_rel')
numsub  = 100;
num_state = 9;

v           = 6.25;
parameters  = randn(1,2);

% 每轮选择套餐载入
if dim == 3
    load('options_3_Dimension.mat')
    food_options_all_round = data;
    num_action = 27;
    trials    = 30;
    num_state_group = 3;
    NumModel = 3;


    

    
    % 所有可能的state载入,这样会有一个state_array在工作区中出现
    load('state_dict._3d.mat')
else
    load('options_4_Dimension.mat')
    food_options_all_round = data;
    num_action = 54;
    trials    = 50;
    num_state_group = 6;
    NumModel = 6;
    
    % 所有可能的state载入,这样会有一个state_array在工作区中出现
    load('state_dict._4d.mat')
end




action_list = cell(9,trials);
score_list = cell(1,trials);
state_list = cell(1,trials);
model_choose_list = cell(1,trials);
all_action_list = zeros(1,trials);


% 初始化参数
nparm   = [3 2 2];
MF          = nan(trials, numsub, NumModel);
NLL         = nan(trials, numsub, NumModel);

subj_data = zeros(trials, numsub, NumModel);

all_state_of_model = zeros(NumModel,trials);

% for i = 1:NumModel
%     m = zeros(nparm(i),1);
%     v = 6.25;
%     prior_RL(i) = struct('mean',m,'variance',v);
%     Theta{i}    = nan(trials, numsub, nparm(i));
% end
% 初始化结束


%%


% for agent =1:numsub
all_agent_action_list = [];
all_agent_score_list = [];
all_model_choose_list = [];
all_agent_full_action_list = table(cell(0,1),'VariableNames',{'full_action_list'});
all_agent_ResponeSignal_list = table(cell(0,1),'VariableNames',{'ResponeSignal_list'});
for agent =1:numsub
    % 在subj_data里同时储存两个model的100个agent的所有信息
    % 计划尝试把每一个值都填充为一个cell
    % cell中包含几个项目：score，action，state
    subj = agent;
    Data= [];

    q_init = .5 * ones(num_state, num_action);
    q_of_model_all =cell(1,NumModel);
    p_init = (1/num_action) * ones(num_state, num_action);
    p_of_model_all = cell(1,NumModel);
    lamda_of_model_all = cell(1,NumModel);
    alpha_of_model_all = cell(1,NumModel);

    for i =1:NumModel
        q_of_model_all{i} = q_init;
        p_of_model_all{i} = p_init;
        lamda_of_model_all{i} = 1/3;
        alpha_of_model_all{i} = 0.1; %初始化学习率
    end
    q_of_model = q_of_model_all;
    p_of_model = p_of_model_all;
    lamda_of_model = lamda_of_model_all;
    alpha_of_model = alpha_of_model_all;

    % 对上面的三个或六个模型，初始化最初始的state
    state_of_all_model = zeros(NumModel,1);
    

    for trial=1: trials
        % 用一种神秘的方式把这一轮的option储存起来，获取options
        options = cell(9,1);
        for option =1: 9
            options{option} = food_options_all_round{option,trial};
        end
        
    
        % 第一轮初始化action
        if trial ==1
            this_trial_action = zeros(num_state_group);

            for i = 1:num_state_group
                state_id = randi([1,num_state]);
                state_of_all_model(i) = state_id;
                chose_action = randi([1,num_action]);
                this_trial_action(i) = chose_action;

            end
            all_state_of_model(:,trial) =  state_of_all_model;

            % 第一轮选择一个action
            action_idx = this_trial_action(randperm(length(this_trial_action), 1));  % 从 A 中随机取 1 个

            % chose_action = randi([1,num_action]);
            % action_idx = chose_action;

        end

        % 假设说这一轮每个模型都已经根据上一轮的q值选好了状态
        % for i = 1:num_state_group
        %     state_id = randi([1,num_state]);
        %     state_of_three_model(i) = state_id;
        % end
        



        full_action = get_action(dim,state_array(action_idx),options);

        action_list(:,trial) = full_action;
        score = reward_func(dim,full_action);

        

        [action_idx,state_of_all_model, q_of_model, p_of_model,which_model_was_choosen,lamda_of_model] = model_RL_9states_002(dim,options,action_idx,state_of_all_model,p_of_model,q_of_model,lamda_of_model,alpha_of_model_all,score);
        choose_action = state_array(action_idx);

        all_action_list(1,trial) =  action_idx;
        action = get_action(dim,choose_action,options);
        action_list(:,trial) = action;
        score = reward_func(dim,action);
        score_list{1,trial} = score;
        state_list{1,trial} = action_idx;
        model_choose_list{1,trial} = which_model_was_choosen;

        
       
        %options = food_options_all_round{0:end,trial};
        p_of_model_last = p_of_model;
        p_of_model_all{trial} = p_of_model;
        q_of_model_all{trial} = q_of_model;
        lamda_of_model_all{trial} = lamda_of_model;
        all_state_of_model(:,trial) =  state_of_all_model;

    end
    all_agent_action_list(agent,:) = all_action_list;
    all_agent_score_list(agent,:) = cell2mat(score_list);
    all_model_choose_list(agent,:) = cell2mat(model_choose_list);

    all_agent_full_action_list.full_action_list{agent,1} = action_list;
    all_agent_ResponeSignal_list.ResponeSignal_list{agent,1}= lamda_of_model_all;
    
end 

%% 
% 假设 choiceTable 是 100x1 的 table，scoreTable 是 100x50 的 double
% 初始化一个新的 cell 数组来存储所有数据
choiceTable = all_agent_full_action_list;
scoreTable = all_agent_score_list;

outputData = table;
%%
for i = 1:100
    % 获取每个 agent 的选择 (9x50 cell array)
    choices = choiceTable{i, 1};  % 这个是 9x50 的 cell 数组
    % choices = choices{1};  % 提取具体的 9x50 数据
    
    % 获取对应的分数 (1x50 double array)
    scores = scoreTable(i, :);  % 这个是 1x50 的 double 数组
    
    % 创建一个空的 cell 数组来存储 agentData
    agentData = cell(50, 1);  % 50 轮
    
    for roundIdx = 1:50
        % 将每轮的 9 个选择与分数作为一个字符串存储
        strArray = cell(9, 1);
        for i = 1:9
            % 将每个数组转换为字符串
            strArray{i} = mat2str(choices{:,i});
        end
        
        % 使用逗号连接每个字符串
        finalStr = strjoin(strArray', ', ');


        agentData{roundIdx} = {finalStr, scores(roundIdx)};
    end
    
    % 将该 agent 的数据存入 outputData 表格中
    outputData.(['Agent' num2str(i)]) = agentData;
end

%%
% 将 cell 数组转换为字符串格式进行保存
finalData = [];
for i = 1:100
    agentData = outputData{i, 1};
    agentDataStr = cell(50, 1);
    
    for roundIdx = 1:50
        % 将每轮的选择和分数合并为字符串形式
        choiceStr = mat2str(agentData{roundIdx}{1});  % 将选择数组转为字符串
        scoreStr = num2str(agentData{roundIdx}{2});  % 将分数转为字符串
        agentDataStr{roundIdx} = [choiceStr, ',', scoreStr];  % 合并为一行
    end
    
    % 将该 agent 的数据合并到 finalData 中
    finalData = [finalData; agentDataStr];
end


% 将整合好的数据保存为 CSV 文件
% 将 cell 数组的内容转换为一个大的矩阵
finalData = vertcat(outputData{:});

% 保存为 CSV 文件



% 将 finalData 展开成一个普通的 cell 数组
finalDataStr = cell(size(finalData));

% for i = 1:size(finalData, 1)
%     % 获取当前 agent 的数据
%     agentData = finalData{i, 1};
% 
%     % 将每轮的选择和分数转化为字符串形式
%     for roundIdx = 1:50
%         choiceStr = mat2str(agentData(roundIdx,:){1});  % 将选择数组转为字符串
%         scoreStr = num2str(agentData{roundIdx}{2});  % 将分数转为字符串
%         finalDataStr{i, roundIdx} = [choiceStr, ',', scoreStr];  % 合并为一行
%     end
% end
% writecell(finalDataStr,'C:\Users\Windows11\Desktop\try.csv')
 