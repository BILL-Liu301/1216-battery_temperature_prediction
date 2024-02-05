clc; clear;
% SOC变化总量
soc_all = 85;
% 时间步骤[1~stamp_all]
stamp_all = 1379;
% 初始电流[0~current_init]
current_init = 249.704728;
% 积分转换系数
K = 2.561927693277232e+03;

% 百分比策略[时间；电流]，只支持预测一个，需要预测的策略请赋0
strategy_percent = [[0.8, 0.9, 1.0];
                    [nan, nan, nan]];

% 将百分比策略转化为实数策略
% strategy_point = [[0, 712, 817, 1004, 1379];
%                   [0, current_init, 137.2, 68.6002, 34.3001]];
strategy_point = round([[0, 0.5; 0, 1.0], strategy_percent] .* [stamp_all - 1; current_init - 0]) + [1; 0];
disp(strategy_point)

% 遗传算法找最优
pop = 200;  % 种群数量
tol = 1e-6; % 允许误差
p1 = 0.006; % 变异率
p2 = 0.5;   % 交叉率
MAX = 100;   % 最大迭代次数
nvars = 3;  % 变量个数
fun = @cal_SOC_integral_fitness;
A = [[1, 0, 0];
     [-1, 1, 0];
     [0, -1, 1]];
b = [current_init; 0; 0];
options = optimoptions('ga', 'ConstraintTolerance', tol, 'PlotFcn', @gaplotbestf, ...
    'MigrationFraction', p1, 'CrossoverFraction', p2, 'Populationsize', pop, 'Display','iter', 'Generations', MAX);
[x, fval] = ga(fun, nvars, A, b, [], [], [0.0, 0.0, 0.0], [250.0, 250.0, 250.0], [], [], options);

[fitness, SOC_integral, strategy_point, strategy] = cal_SOC_integral(x);
result = [transpose(strategy), transpose(SOC_integral) + 10];
