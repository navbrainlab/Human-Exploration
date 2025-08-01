%% 假设文件名是 'mydata.csv'
clear;
clc;
opts = detectImportOptions('options_4_Dimension.csv', 'NumHeaderLines', 0);
opts = setvartype(opts, 'char');  % 先都按文本读取，防止数组格式出错
T = readtable('options_4_Dimension.csv', opts);

T = T(2:end,:);

nOptions = height(T);
nRounds = width(T);
data = cell(nOptions, nRounds);

for round = 1:nRounds
    for i = 1:nOptions
        
        array_str = T{i, round}{1};     % 从 cell 中取出字符串：'[1, 1, 1]'
        array_num = str2num(array_str); % 转换成数值数组    → [1 1 1]
        data{i,round} = array_num;  % 被试名

        
        % 将字符串 "[1,2,3]" 转换为真正的数值数组
        % data{i}.choices  = str2num(T.choices{i});    %#ok<ST2NM>
        % data{i}.outcomes = str2num(T.outcomes{i});   %#ok<ST2NM>
    end
end


save('C:\Users\Windows11\Documents\MATLAB\RL-modelling\jilab\options_4_Dimension.mat', 'data');

%%
clear;
clc;
opts = detectImportOptions('choice_reward_pair.csv', 'NumHeaderLines', 0);
opts = setvartype(opts, 'char');  % 先都按文本读取，防止数组格式出错
T = readtable('choice_reward_pair.csv', opts);
T = T(:,3:end);

nOptions = height(T);
nRounds = width(T);
data = cell(nOptions, nRounds);

% 假设该列是 cell 数组的形式
strs = T.blocks;

% 先转换这个列
numLists = cell(height(T), 1);
for i = 1:height(T)
    cleanStr = erase(erase(strs{i}, '('), ')');
    strArray = split(cleanStr, ',');
    numArray = str2double(strArray);
    numLists{i} = numArray;
end

% 替换掉原来的列
T.blocks = numLists;

% 其他列都转为 double
otherVarNames = T.Properties.VariableNames;
otherVarNames(strcmp(otherVarNames, 'blocks')) = [];  % 去掉已处理的列

for i = 1:length(otherVarNames)
    varName = otherVarNames{i};
    % 假设它们原来是 cell array of char，需要转成数字
    T.(varName) = str2double(T.(varName));
end
save('C:\Users\Windows11\Documents\MATLAB\RL-modelling\jilab\choice_reward_pair.mat', 'T')