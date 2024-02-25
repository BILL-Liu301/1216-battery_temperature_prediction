% 读取模组内所有组电芯的数据
% 数据清空
clc; clear;
figure;

% 加载数据，[1: voltage, 2: ntc_max, 3: ntc_min, 4: temperature_max]
load('results\常温充电.mat')
soc = origin(1, :, 5)';
temperature_pre = pre_mean(:, :, 4)';
temperature_ref = ref_mean(:, :, 4)';

% 绘制三维温度变化图
x = 0:4:96;
y = ones(size(x));
pbar = waitbar(0, '正在动态绘图...');
for seq = round(linspace(1, size(temperature_pre, 1), 100))
    
    % 生成网格，进行插值
    % pre
    z_pre = double(temperature_pre(seq, :));
    % ref
    z_ref = double(temperature_ref(seq, :));
    
    % 绘图
    clf
    
    % 三维温度视图
    subplot(4, 2, [1, 3, 5])
    b_pre = bar3(x, z_pre);
    hold on
    colormap(b_pre, [0 0 1; 1 0 0]); % blue to red colormap 
    colorbar

    % 进度条
    waitbar(seq/size(temperature_pre, 1), pbar, ['绘制中...', num2str(round(100*seq/size(temperature_pre, 1), 1)),'%'])
    pause(0.01)
end


