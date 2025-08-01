load('choice_reward_pair.mat');
dim = 3;

choice_reward_pair = T;
x_list = choice_reward_pair.X_coord;
score_list = choice_reward_pair.value;
colors = [0 0 1; 0 0.79 0.34; 1 0.6 0.07; 0.7 0.13 0.13];
if dim==3
    trial_length =30;
else
    trial_length = 50;
end

for i = 1:100
% for i = 1: height(all_agent_full_action_list) %每个agent
    this_agent_action = all_agent_full_action_list(i,:); %所有trial的action
    this_agent_action = this_agent_action.full_action_list{1};
    all_trial_type = [];
    all_h = [];
    for j = 1: length(this_agent_action) %每轮trial遍历

        this_trial_action = this_agent_action(:,j);
        all_food_type = [];
        this_trial_h = [];
        for food = 1:dim %每个食物的排列方式
            this_food_arrange = cellfun(@(x) x(food),this_trial_action);
            V = reshape(this_food_arrange,3,[]);
            v_sorted = sort(V,1);
            this_food_arrange = reshape(v_sorted,[],1);

            for h = 1:height(choice_reward_pair)
                thisVec = choice_reward_pair.blocks{h};
                
                if isequal(thisVec, this_food_arrange) 
                    matchIndex = h;
                    break;
                end
            end
            this_food_type = choice_reward_pair.choices_category(h);
            all_food_type(end+1) = this_food_type;
            this_trial_h(end+1) = h;
        end
        all_trial_type = [all_trial_type;all_food_type];
        all_h = [all_h;this_trial_h];
    end
    f1 = figure;
    hold on;
    ax = gca;
    ax.Layer = 'top';
    
    rectangle('Position',[0,0,46.6700,trial_length+2],'FaceColor',[0,0.79,0.34],'EdgeColor','none'); %122
    rectangle('Position',[101.48,0,145.19-101.48,trial_length+2],'FaceColor',[1,0.92,0.8],'EdgeColor','none'); %222
    rectangle('Position',[223.33,0,1,trial_length+2],'FaceColor',[0.53,0.15,0.45],'EdgeColor','none'); %111
    rectangle('Position',[291.11,0,357.78-291.11,trial_length+2],'FaceColor',[1,0.75,0.8],'EdgeColor','none'); %223
    rectangle('Position',[389.26,0,459.26-389.26,trial_length+2],'FaceColor',[0.69,0.88,0.9],'EdgeColor','none'); %222
    rectangle('Position',[500-27.593,0,100,trial_length+2],'FaceColor',[0.75,0.75,0.75],'EdgeColor','none');
    
    score = all_agent_score_list(i,:);
    plot(score+450,1:trial_length,'DisplayName','Score','Color',[0.69,0.09,0.12]);
    
    
    for food =1: width(all_h)
        all_trial_idx = all_h(:,food);
        x = x_list(all_trial_idx);
        y = score_list(all_trial_idx);
        plot(x,1:numel(x),'Color',colors(food,:),'DisplayName',['dim',num2str(food)]);
        
    end
    % legend;

        % for trial = 1:length(all_h)
    ylim([0,trial_length+2]);
    xlim([-30,600]);
    set(gca,'XTick',[0,50-27.593,150-27.593 ,250-27.593,350-27.593,450-27.593,500-27.593,550-27.593,600-27.593]);
    set(gca,'XTickLabel',{'0','122','222','111','223','333','0','Points','100'});
    % xticks([0,50-27.593,150-27.593 ,250-27.593,350-27.593,450-27.593,500-27.593,550-27.593,600-27.593])
    % xticklabels({['0','122','222','111','223','333','0','Points','100']})
    filename = fullfile('C:\Users\Windows11\Desktop\3D_ARL_data', [num2str(i) '.png']);
    frame = getframe(f1);
    imwrite(frame.cdata,filename);
    close all
end

%% 画平均分数变化
mean_vals = mean(all_agent_score_list, 1);         % 按列求均值，结果是 1×30
std_vals  = std(all_agent_score_list, 0, 1);       % 按列求标准差，结果是 1×30
ste = std_vals/sqrt(100);
hold on;
xlim([0,trial_length+1]);

set(gca,'YLim',[70,100]);
 
upper = mean_vals + ste;
lower = mean_vals - ste;

x = 1:size(all_agent_score_list, 2);               % x = 1 到 30
% 画误差带（先画）
fill_x = [x, fliplr(x)];
fill_y = [upper, fliplr(lower)];

fill(fill_x, fill_y, [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.4);

plot(x,mean_vals)
% errorbar(x, mean_vals, ste); % 带误差棒的折线图
xlabel('Round');
ylabel('Score');
title('Column-wise Mean with Error Bars');
% grid on;

%% 画expected value & responsibility 柱状图

data_of_all_model = {zeros(100,1);zeros(100,1);zeros(100,1)};
if dim ==3
    model_num = 3;
    xlabel = {'AbRL_1_2','AbRL_1_3','AbRL_2_3'};
    set(gca,'XTick',[1,2,3]);
else
    model_num = 6;
    xlabel = {'AbRL_1_2','AbRL_1_3','AbRL_2_3','AbRL_1_4','AbRL_2_4','AbRL_3_4'};
    set(gca,'XTick',[1,2,3,4,5,6]);
end




for agent = 1:100
    this_agent_all_trial_signal = all_agent_ResponeSignal_list{agent,1};
    this_agent_all_trial_signal = this_agent_all_trial_signal{1};
    %统计最后一个trial的数据
    % for trial = 1:length(this_agent_all_trial_signal)
    trial = length(this_agent_all_trial_signal);
    this_trial_three_model_signal = this_agent_all_trial_signal(:,trial);
    % this_trial_three_model_signal = this_trial_three_model_signal{1};
    for model = 1:model_num
        this_model_signal = this_trial_three_model_signal{model};
        data_of_all_model{model}(agent) = this_model_signal;
    end
end

mean_data_of_all_model = zeros(1,model_num);
ste_data_of_all_model = zeros(1,model_num);
for model = 1:model_num
    mean_of_this_model = mean(data_of_all_model{model});
    mean_data_of_all_model(model) = mean_of_this_model;
    ste_data_of_all_model(model) = std(data_of_all_model{model})/10;
end
colors = [200,65,68; 200,132,174; 249,199,179; 100,190,190; 135,207,235; 160,102,212]/255;

hold on

y = mean_data_of_all_model;
for i = 1:model_num
    b = bar(i,y(i),0.9,'stacked');
    alpha(0.1);
    set(b(1),'facecolor',colors(i,:));
    scatter([i]*length(y(i)),data_of_all_model{i},50,colors(i,:));
end
lower = y - ste_data_of_all_model;
upper = y + ste_data_of_all_model;
errorbar(1:model_num, y, lower,upper,'Marker','none','LineStyle','none','Color','black');

box on


set(gca,'XtickLabel',xlabel);
ylabel('λ (Responsibility)')








