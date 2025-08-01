%% settings
clear;
bl = [0 0.4470 0.7410];
or = [235 161 51]/255;
ye = [242 206 108]/255;
gr = [131 157 68]/255;
red = [225 112 110]/255;
dark_bl = [48 151 164]/255;
pur = [0.4940 0.1840 0.5560];
dark = [77 77 79]/255;
light_dark = [100 100 103]/255;
brown = [171 104 87]/255;
link_w = 2.5;
mark_size = 12;
colors = [bl;gr;or;red;pur;light_dark;dark_bl;brown;ye];
markers = {'>', '+', 'diamond', 'o', 'v', '*', 'square', '^','x'};

rng(0)
%% Offline Evaluation
figure;
data = readtable('uma-pre-adapt-offline-eval.csv','VariableNamingRule', 'preserve');
loc_colors=[bl;or;red;dark_bl;brown;light_dark;gr;pur];
x = data{:, 1};
y = data{:, 2:end};
for i = 1:size(y, 2)
    semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',markers{i});
    hold on;
end
set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(2:end), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best', ...
        'FontSize',16)

xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;

%% Online Eval
data = readtable('umi-post-adapt-online-eval.csv','VariableNamingRule', 'preserve');
loc_colors = [bl;red;or;dark_bl;brown;light_dark;gr;pur];

x = data{:, 1};
select_cols=[1,2,3,5,6,7,8,9]+1;
y = data{:,select_cols};
for i = 1:size(y, 2)
    semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',markers{i});
    hold on;
end
set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(select_cols), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best', ...
        'FontSize',16)

xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;


%% Forgetting check

figure;
data = readtable('uma_forget_check.csv','VariableNamingRule', 'preserve');
loc_colors=[bl;red;light_dark;gr;pur];
loc_markers={'+','diamond','*', 'square', '^'};
x = data{:, 1};
y = data{:, 2:end};

for i = 1:size(y, 2)
    semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',loc_markers{i});
    hold on;
end

set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(2:end), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best',...
        'FontSize',16)


xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;





