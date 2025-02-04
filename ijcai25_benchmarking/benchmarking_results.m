%%% code to plot computation times for symbolic regret synthesis 
clc; clear all; close all;

set(groot,'defaulttextInterpreter','latex');

%%%% set Plot_svg = 0 to print in png and
%%%% set Plot_svg = 1 to print in svg format
PLOT_SVG = 1;

line_thick = 2;
fontsize = 12;

budget = [20, 24, 28, 32, 36, 40, 44];
spec1 = [15.42, 20.45, 28.03, 30.60, 39.21, 41.55, 50.53];
spec2 = [24.01, 32.42, 45.83, 57.60, 74.17, 81.11, 100.77];
spec3 = [32.31, 47.61, 65.56, 84.64, 110.69, 130.82, 153.34];
spec4 = [44.58, 62.57, 85.57, 118.03, 145.11, 170.67, 200.82];

f = figure();
plot(budget, spec1, '-o', 'LineWidth', line_thick);
hold on;
plot(budget, spec2, '-o', 'LineWidth', line_thick);
hold on;
plot(budget, spec3, '-o', 'LineWidth', line_thick);
grid on
hold on;
plot(budget, spec4, '-o', 'LineWidth', line_thick);

xticks(budget)
xlabel('Budget ($\mathcal{B}$)', 'FontSize', fontsize)
ylabel('Computation Time (s)', 'FontSize', fontsize)
legend('\phi_1', '\phi_2', '\phi_3', '\phi_4', 'Location', 'northwest')
% legend('Explicit', 'Partitioned', 'Location', 'Best')
% title('Budget vs Total Computation Time')

if PLOT_SVG == 1
    saveas(f,[pwd, '/compute_bench.svg'])
elseif PLOT_SVG == 0
    saveas(f,[pwd, '/compute_bench.png'])
end

return