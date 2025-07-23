function [action,state_of_three_model, q_of_3model,p_of_3model] = model_RL_9states(options,last_action,state_of_three_model,p_of_3model,q_of_3model,lamda_of_3model,alpha_of_3model, last_score)

num_state = 9;
num_state_group = 3;

% 所有可能的state载入,这样会有一个state_array在工作区中出现
load('state_dict.mat');

this_trial_action_list = zeros(3);
% 感觉action可以参照我们的hypothesis，设置为认为哪个维度的哪个食物重要
% 否则没完没了了，
% 但是如果局限于1个维度里有1个重要，像我们贝叶斯那样的话，这也是不太严谨的
% 现在试试每轮action
num_action = 27;
parameters = [0.1, 0.01];



for model = 1:length(q_of_3model)
    nd_alpha    = alpha_of_3model{model}; % normally-distributed alpha
    alpha       = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)
    
    nd_beta     = parameters(2);
    beta        = exp(nd_beta);
    p_all = zeros(num_state,num_action);

    % 将action和state对比
    % 并计算Q值
    state = state_of_three_model(model);

    q = q_of_3model(model);
    q = q{1};
    q_convert = q;
    delta1  = last_score/100 - q_convert(state, last_action); % prediction error
    q_convert(state, last_action) = q_convert(state, last_action) + (alpha*delta1); 
    
    % for a = 1:num_action
    %     if a ~= last_action  % 对未被选择的动作
    %         delta = (1 - last_score) - q_convert(state, a);  % 或者其他形式，比如 -reward - q(...)
    %         q_convert(state, a) = q_convert(state, a) + alpha * delta;
    %     end
    % end
    
    q_of_3model{model} = q_convert;

    qvals = q_convert(state, :);            % 取当前状态下所有动作的 Q 值
    qvals = beta * qvals;           % 按照 beta 缩放
    qvals = qvals - max(qvals);     % 防止溢出（数值稳定）
    
    p = exp(qvals) / sum(exp(qvals));  % Softmax 转换，得到动作概率分布
    p_this = p;
    p_of_3model{model} = p_this;

    action = randsample(length(p_this),1,true,p_this);
    % action = randi([1,9]);
    this_trial_action_list(model) = action; 
    
    state = randi([1,9]);
    state_of_three_model(model) = state;
    
end

% 选择action
% action_id = randi([1,3]);
% action = this_trial_action_list(action_id);

% 选择一个action,通过比较(?)
chose_action = 1;
best_reward = 0;

% disp(this_trial_action_list);
% disp(length(this_trial_action_list));
for i = 1:length(this_trial_action_list)
    choose_state_idx = this_trial_action_list(i);
    choose_state = state_array(choose_state_idx);
    % disp(choose_state_idx);
    
    action = get_action(choose_state,options);

    
    reward = reward_func(action);

    if reward >best_reward
        chose_action = this_trial_action_list(i);
        best_reward=reward;
    end
end

action = chose_action;

end