function [fitness, SOC_integral, strategy_point, strategy] = cal_SOC_integral(current)
    K = 2.561927693277232e+03;
    current_init = 249.704728;
    % 实例化策略，并积分
    strategy = [[]; []];
    strategy_point = [[0.0, 690.0; 
                       0.0, current_init], ... 
                      [1103.0, 1241.0, 1379.0;
                      current]];
    for stra = 1:(size(strategy_point, 2) - 1)
        stamp_0 = strategy_point(1, stra) + 1;
        stamp_1 = strategy_point(1, stra + 1);
        stamp = stamp_0:stamp_1;
        current = strategy_point(2, stra + 1) * ones(size(stamp));
        strategy = [strategy, [stamp; current]];
    end
    
    SOC_integral = zeros(1, size(strategy, 2));
    for t=2:size(strategy, 2)
        SOC_integral(t) = (strategy(2, t - 1) + strategy(2, t)) * (strategy(1, t) - strategy(1, t - 1)) / 2 + SOC_integral(t - 1);
    end
    SOC_integral = SOC_integral / K;
    % fitness = 1 / (abs(SOC_integral(1, end) - 85) + 1e-6);
    fitness = abs(SOC_integral(1, end) - 85);
end