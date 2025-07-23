function [action] = get_action(dim,choose_state,options)
score_of_set_list = zeros(length(options),1);
if dim == 3
    state_choice = {choose_state.one,choose_state.two,choose_state.three};
else
    state_choice = {choose_state.one,choose_state.two,choose_state.three,choose_state.four};
end
for option_id =1:length(options) 
    option = options(option_id);
    option = option{1};
    score_of_set = 0;
    for food_id = 1:length(option) 
        % food是这个option中第几个维度选了第几个食物
        food = option(food_id);
        food_state = state_choice{food_id};
        if food_state(food)==1
            score_of_set = score_of_set+1;
        end
    end
    score_of_set_list(option_id) = score_of_set;

    [~, idx] = sort(score_of_set_list, 'descend');   % 得到按从大到小排序的索引
    action = options(idx);               % 用这个索引重排 cell


end


