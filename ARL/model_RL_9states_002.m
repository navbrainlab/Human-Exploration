function [action,state_of_all_model, q_of_model,p_of_model,which_model_was_choosen,lamda_of_model] = model_RL_9states_001(dim,options,last_action,state_of_all_model,p_of_model,q_of_model,lamda_of_model,alpha_of_model, last_score)


this_trial_action_list = zeros(3);
this_trial_q_list = zeros(3);
% 感觉action可以参照我们的hypothesis，设置为认为哪个维度的哪个食物重要
% 否则没完没了了，
% 但是如果局限于1个维度里有1个重要，像我们贝叶斯那样的话，这也是不太严谨的
% 现在试试每轮action
parameters = [0.1, 0.01];



for model = 1:length(q_of_model)
    nd_alpha    = alpha_of_model{model}; % normally-distributed alpha

    % alpha = nd_alpha;
    alpha       = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)s
    
    nd_beta     = parameters(2);
    beta        = exp(nd_beta);
    % 将action和state对比
    % 并计算Q值
    state = state_of_all_model(model);

    q = q_of_model(model);
    q = q{1};
    q_convert = q;
    delta1  = last_score/100 - q_convert(state, last_action); % prediction error
    q_convert(state, last_action) = q_convert(state, last_action) + (alpha*delta1); 
    
    % for a = 1:num_action
    %     if a ~= last_action  % 对未被选择的动作
    %         delta = (1 - last_score/100) - q_convert(state, a);  % 或者其他形式，比如 -reward - q(...)
    %         q_convert(state, a) = q_convert(state, a) + alpha * delta;
    %     end
    % end
    
    q_of_model{model} = q_convert;

    qvals = q_convert(state, :);            % 取当前状态下所有动作的 Q 值
    % qvals = beta * qvals;           % 按照 beta 缩放
    % qvals = qvals - max(qvals);     % 防止溢出（数值稳定）
    
    p = exp(qvals) / sum(exp(qvals));  % Softmax 转换，得到动作概率分布
    p_this = p;
    p_of_model{model} = p_this;
    % [max_val, action] = max(p_this);
    action = randsample(length(p_this),1,true,p_this);
    % action = randi([1,9]);
    this_trial_action_list(model) = action; 
    this_trial_q_list(model) = qvals(action);
    
    state = randi([1,9]);
    state_of_all_model(model) = state;
    
end

% 选择action
% action_id = randi([1,3]);
% action = this_trial_action_list(action_id);

% 选择一个action,通过比较(?)
chose_action = 1;
best_q = 0;



for i = 1:length(this_trial_action_list)
    this_q = this_trial_q_list(i);
    if this_q > best_q
        chose_action = this_trial_action_list(i);
        which_model_was_choosen = i;
        best_q = this_q;
    end
end
for m = 1:length(lamda_of_model)
    if m == which_model_was_choosen
        lamda_of_model{m} = lamda_of_model{m}*1.1;
    else
        lamda_of_model{m} = lamda_of_model{m}*0.9;
    end
end

action = chose_action;

end