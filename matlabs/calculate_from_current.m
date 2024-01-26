% 初始化
clc; clear;
figure

% 基础参数
path_xlsx = "../datas/T9M仿真温度数据_to华工.xlsx";

% 读取xlsx数据
data_xlsx = readtable(path_xlsx, "Sheet", "常温充-电流电压SOC", "VariableNamingRule", "preserve");

% 从xlsx提取数据
stamp = data_xlsx.("时间")(~isnan(data_xlsx.("时间")));
I = data_xlsx.("电流")(~isnan(data_xlsx.("电流")));
SOC = data_xlsx.("SOC")(~isnan(data_xlsx.("SOC")));

% 电流积分
SOC_integral = zeros(size(I));
for t=2:size(stamp)
    SOC_integral(t) = (I(t - 1) + I(t)) * (stamp(t) - stamp(t - 1)) / 2 + SOC_integral(t - 1);
end
% 重点系数
K = mean(SOC_integral(2:end) ./ (SOC(2:end) - SOC(1)));
SOC_integral = SOC_integral / K;

subplot(2, 1, 1)
p1 = plot(I, 'r-', 'LineWidth', 1);
hold on
legend(p1, '电流')
legend('Location', 'southeast')

subplot(2, 1, 2)
p1 = plot(SOC - SOC(1), 'b.', 'LineWidth', 1);
hold on
p2 = plot(SOC_integral, 'b--', 'LineWidth', 1);
hold on
legend([p1, p2], 'SOC-真实', 'SOC-积分')
legend('Location', 'southeast')
