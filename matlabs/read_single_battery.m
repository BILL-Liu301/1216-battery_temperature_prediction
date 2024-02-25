% 初始化
clc; clear;
figure

%% 读取xlsx内数据，分析不同电芯的不同点温度随时间变化趋势
% 基础参数
path_xlsx = "../datas/real/T9M充电数据汲内阻-to华工.xlsx";
Mx = "02-";

% 读取xlsx数据
data_xlsx = readtable(path_xlsx, "Sheet", "常温充电", "VariableNamingRule", "preserve");

% 设定需要分析的电芯，并按顺序设定电信测温点的位置
% 内：两模组之间。外：两模组之外
%       上
%     1   2
% 内 3  4   5 外
%     6   7
%       下

title_ = 'M2-100';
%temperature_id = {"05", "02", "07", "04", "01", "06", "03"}; % M1-1
%temperature_id = {"09", "12", "08", "11", "14", "10", "13"}; % M1-28
%temperature_id = {"16", "19", "15", "18", "21", "17", "20"}; % M1-56
%temperature_id = {"23", "26", "22", "25", "28", "24", "27"}; % M1-100
%temperature_id = {"05", "02", "07", "04", "01", "06", "03"}; % M2-1
%temperature_id = {"09", "12", "08", "11", "14", "10", "13"}; % M2-28
%temperature_id = {"16", "19", "15", "18", "21", "17", "20"}; % M2-56
temperature_id = {"23", "26", "22", "25", "28", "24", "27"}; % M2-100

temperature_id_pos.x = [-1, 1, -2, 0, 2, -1, 1];
temperature_id_pos.y = [1, 1, 0, 0, 0, -1, -1];
temperature_measure =  [];
for id = temperature_id
    temperature_measure = [temperature_measure, data_xlsx.(Mx + string(id))(~isnan(data_xlsx.(Mx + string(id))))];
end

% 动态绘图
X = temperature_id_pos.x;
Y = temperature_id_pos.y;
temperature_measure_mean = mean(temperature_measure, 2);
T = data_xlsx.("时间[s]")(~isnan(data_xlsx.("时间[s]")));
I = data_xlsx.("电流")(~isnan(data_xlsx.("电流")));
pbar = waitbar(0, '正在动态绘图...');
for seq = round(linspace(1, size(temperature_measure, 1), 10))

    % 提取每一时刻的Z数据
    Z = temperature_measure(seq, :);
    
    % 生成网格，进行插值
    [Xn, Yn] = meshgrid(min(X):0.1:max(X), min(Y):0.1:max(Y));
    Zn = griddata(X, Y, Z, Xn, Yn);
    
    % 绘图
    clf
    
    % 三维温度视图
    subplot(2, 1, 1)
    surf(Xn, Yn, Zn)
    hold on
    for p = 1:size(Z, 2)
        plot3([X(1, p), X(1, p)], [Y(1, p), Y(1, p)], [0, Z(1, p)], 'k--', 'LineWidth', 2)
        text(X(1, p), Y(1, p), Z(1, p)+0.5, temperature_id{p})
    end
    text(-2, 0, min(min(temperature_measure))+1, '内')
    text(2, 0, min(min(temperature_measure))+1, '外')
    title(title_)
    xlim([-2, 2])
    ylim([-1, 1])
    zlim([min(min(temperature_measure))-1, max(max(temperature_measure))+1])
    zlabel('温度/℃')
    colorbar
    view(2, 10)
    
    % 二维电流视图
    subplot(2, 1, 2)
    yyaxis left
    p1 = plot(T(1:seq), I(1:seq), 'b-', 'LineWidth', 2);
    span = max(I) - min(I);
    xlim([T(1), T(end)])
    ylim([min(I) - span * 0.1, max(I) + span * 0.1])
    xlabel('时间', 'Color', 'b')
    ylabel('电流/A', 'Color', 'b')

    yyaxis right
    for point = 1:1:size(temperature_measure, 2)
        p2 = plot(T(1:seq), temperature_measure(1:seq, point), 'r-', 'LineWidth', 1);
        hold on
    end
    span = max(max(temperature_measure)) - min(min(temperature_measure));
    ylim([min(min(temperature_measure)) - span * 0.1, max(max(temperature_measure)) + span * 0.1])
    ylabel('各测点温度/℃', 'Color', 'r')

    grid on
    legend([p1, p2], '电流', '各测点温度')
    legend('Location', 'southeast')

    % 进度条
    waitbar(seq/size(temperature_measure, 1), pbar, ['绘制中...', num2str(round(100*seq/size(temperature_measure, 1), 1)),'%'])
    pause(0.01)

end

