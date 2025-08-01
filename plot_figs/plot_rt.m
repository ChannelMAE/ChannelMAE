%% settings
clear;
% bl = [111 140 189]/255;
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
link_width = 2.5;
mark_size = 12;
colors = [bl;gr;or;red;pur;light_dark;dark_bl;brown;ye];
markers = {'>', '+', 'diamond', 'o', 'v', '*', 'square', '^','x'};
rng(0)

%% Dist Shift
data = readtable('new_results/dist-shift.csv','VariableNamingRule', 'preserve');
loc_colors=[red;light_dark;gr;pur];
loc_markers={'diamond','*', 'square', '^'};

x = data{:, 1};
y = data{:, 2:end-1};
for i = 1:size(y, 2)
        semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',loc_markers{i});
        hold on;
end

xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;
set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(2:end), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best', ...
        'FontSize',16)


%% Training loss
data = readtable('loss_two_tasks.csv','VariableNamingRule', 'preserve');
x = data{1:557,1};
main = data{1:557,2};
aux=data{1:557,3};
num_avg=30;
avg_main = runningAverage(main, num_avg);
avg_aux = runningAverage(aux, num_avg);

figure;
h1 = plot(x, aux, 'Color', [bl,0.2], 'LineWidth', 1, 'MarkerSize', mark_size); % No legend
hold on;
h2 = plot(x, avg_aux, 'Color', bl, 'LineWidth', 2.5, 'MarkerSize', mark_size); % Include in legend
hold on;
h3 = plot(x, main, 'Color', [or,0.2], 'LineWidth', 1, 'MarkerSize', mark_size); % No legend
hold on;
h4 = plot(x, avg_main, 'Color', or, 'LineWidth', 2.5, 'MarkerSize', mark_size); % Include in legend

% Add legend ONLY for h2 (avg_aux) and h4 (avg_main)
legend([h2, h4], {'Main-task (channel estimation)','SSL-task'});
yticks([0 0.01 0.02 0.03 0.04 0.05]);
xlabel("Online Batch")
ylabel("Online Training Loss");

set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;


%% Plot offline-eval
figure;

% Load the CSV file
data = readtable('new_results/rt4-pre-adapt-offline-eval.csv','VariableNamingRule', 'preserve');
loc_colors=[bl;or;red;dark_bl;brown;light_dark;gr;pur];

x = data{:, 1};
y = data{:, 2:end-1};
for i = 1:size(y, 2)
        semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',markers{i});
        hold on;
end

xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;
set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(2:end), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best', ...
        'FontSize',16)


%% Online Eval.
figure;
data = readtable('new_results/rt2-post-adapt-online-eval.csv','VariableNamingRule', 'preserve');
loc_colors = [bl;red;or;dark_bl;brown;light_dark;gr;pur];
select_cols=[1,2,3,5,6,7,8,9]+1;
x = data{:, 1};
y = data{:, select_cols};

for i = 1:size(y, 2)
        semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',markers{i});
        hold on;
end
xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 15 18],'DefaultLineLineWidth',2.5);
grid on;
box on;

set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(select_cols), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best', ...
        'FontSize',16)


%% Forgetting check
data = readtable('new_results/on-rt4-forgetting-check.csv','VariableNamingRule', 'preserve');
loc_colors=[bl;red;light_dark;gr;pur];
loc_markers={'+','diamond','*', 'square', '^'};

x = data{:, 1};
y = data{:, 2:end};


for i = 1:size(y, 2)
        semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',loc_markers{i});
        hold on;
end

xlabel("SNR (dB)")
ylabel("Channel Estimation MSE");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;

set(gca, 'XTick', [0 5 10 15 20])
legend( data.Properties.VariableNames(2:end), ...
        'Interpreter','none', ...   % don't try to parse underscores as subscripts
        'Location','best', ...
        'FontSize',16)


%% Continual plots
data = readtable('new_results/continual.csv','VariableNamingRule', 'preserve');
x = data{:,1};
mae = data{:,2};
super=data{:,3};
num_avg=30;
avg_mae = runningAverage(mae, num_avg);
avg_super = runningAverage(super, num_avg);

figure;
h1 = semilogy(x, super, 'Color', [or,0.1], 'LineWidth', 1, 'MarkerSize', mark_size); % No legend
hold on;
h2 = semilogy(x, avg_super, 'Color', or, 'LineWidth', 2.5, 'MarkerSize', mark_size); % Include in legend
hold on;
h3 = semilogy(x, mae, 'Color', [bl,0.1], 'LineWidth', 1, 'MarkerSize', mark_size); % No legend
hold on;
h4 = semilogy(x, avg_mae, 'Color', bl, 'LineWidth', 2.5, 'MarkerSize', mark_size); % Include in legend

% Add legend ONLY for h2 (avg_super) and h4 (avg_mae)
legend([h2, h4], {'Supervised', 'ChannelMAE'});

xlabel("Online Batch")
ylabel("Main-Task (Channel Est.) Loss");

set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;


%% Compute Continual MSE Gain
data = readtable('new_results/epoch_continual.csv','VariableNamingRule', 'preserve');
g1=10*log10(data{1,2}/data{2,2});
g2=10*log10(data{3,2}/data{4,2});
g3=10*log10(data{5,2}/data{6,2});
g4=10*log10(data{7,2}/data{8,2});
disp([g1,g2,g3,g4]);


%% Per Epoch Continual
data = readtable('new_results/epoch_continual.csv','VariableNamingRule', 'preserve');
x = data{:,1};
mae = data{:,2};
super=data{:,3};
figure;
semilogy(x, super, 'Color', or, 'LineWidth', 2.5, 'MarkerSize', mark_size)
xlabel("Batch Step")
ylabel("Main-Task (Channel Est.) Loss");
set(gca,'Fontname','times new Roman','FontSize',20,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;

