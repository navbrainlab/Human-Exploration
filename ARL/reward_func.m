function [reward] = reward_func(dim,action)

row_weight = [10,5,1];
food_1 = [10,5,1];
food_2 = [10,5,1];
food_3 = [0,0,0]; % 3D
food_4 = [0,0,0];
if dim ==3
    all_food_score = {food_1,food_2,food_3};
else
    all_food_score = {food_1,food_2,food_3,food_4};
end

action_score = 0;
for a_idx = 1:length(action)
    this_set = action(a_idx);
    this_set = this_set{1};
    row = ceil(a_idx/3);

    set_score = 0;

    for food = 1:width(this_set)

        food_score = all_food_score{food};
        this_food_score = food_score(this_set(food));
        set_score = set_score + this_food_score;
    end


    set_score = row_weight(row)*(set_score);
    action_score = action_score + set_score;
end

reward = (action_score - 350) * 90 / 324 + 10;

if reward ~= 0 && reward ~= 100
    noise = randi([-2,2]);
    reward = reward + noise;
end



