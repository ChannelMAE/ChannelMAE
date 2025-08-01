%% Ablation_studies
%% settings
clear;
% bl = [111 140 189]/255;
bl = [0    0.4470    0.7410];
or = [235 161 51]/255;
ye = [242 206 108]/255;
gr = [131 157 68]/255;
red = [225 112 110]/255;
dark_bl = [48 151 164]/255;
pur = [0.4940    0.1840    0.5560];
dark = [77 77 79]/255;
light_dark = [100 100 103]/255;
brown = [171 104 87]/255;
link_w = 2.5;
mark_size = 12;
colors = [bl;gr;or;red;pur;light_dark;dark_bl;brown;ye];
markers = {'>', '+', 'diamond', 'o', 'v', '*', 'square', '^','x'};
rng(0)

%% Model Arch Variation
% pretrain@offline env.
en1_dm1_da1_offline = [0.04606308788061142	0.01436044555157423	0.005958180874586105	0.002917643636465073	0.0020958271343261];
en1_dm2_da2_offline = [0.0488969162106514	0.0150280799716711	0.005829410161823034	0.002607204718515277	0.001876379596069455];
en1_dm4_da2_offline = [0.05272422358393669	0.01464245840907097	0.004593867342919111	0.001948320772498846	0.001448263297788799];
en2_dm4_da2_offline = [0.05323053523898125	0.01390235405415297	0.004637439735233784	0.002055136486887932	0.001560472417622805];

% adapt@online env.
en1_dm1_da1_online = [0.04786825552582741	0.01729638315737247	0.007815010845661163	0.004459553863853216	0.004268412012606859];
en1_dm2_da2_online = [0.05620404705405235	0.01869633980095387	0.0072494694031775	0.004572188016027212	0.004000585060566664];
en1_dm4_da2_online = [0.05638652667403221	0.01563427411019802	0.00568010238930583	0.003263447666540742	0.002564249560236931];
en2_dm4_da2_online = [0.05493746697902679	0.01659626886248589	0.006918085739016533	0.004603054840117693	0.004050506744533777];

snr_index=4; % @15dB
offline_vals = [en1_dm1_da1_offline(snr_index), en1_dm2_da2_offline(snr_index), en1_dm4_da2_offline(snr_index), en2_dm4_da2_offline(snr_index)];
online_vals  = [en1_dm1_da1_online(snr_index),  en1_dm2_da2_online(snr_index),  en1_dm4_da2_online(snr_index),  en2_dm4_da2_online(snr_index)];

% Plot grouped bar chart
figure;
h_bar = bar([offline_vals; online_vals]);

% Set bar colors and border width
loc_colors=[bl;gr;red;light_dark];
h = gca;
for i = 1:4
    h.Children(5-i).FaceColor = loc_colors(i,:);
    h.Children(5-i).LineWidth = 1.5;  % Set border width
end

ylim([0.0015 0.005])
yticks
set(gca, 'XTick', 1:2, 'XTickLabel', {'Offline-Pretrained','Online-Adapted'});
legend('$N_\mathrm{e}=1,N_\mathrm{dm}=1,N_\mathrm{ds}=1$','$N_\mathrm{e}=1,N_\mathrm{dm}=2,N_\mathrm{ds}=2$', ...
    '$N_\mathrm{e}=1,N_\mathrm{dm}=4,N_\mathrm{ds}=2$','$N_\mathrm{e}=2,N_\mathrm{dm}=4,N_\mathrm{ds}=2$', ...
    'Interpreter', 'latex', ...
    'FontSize',16);
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gca, 'YScale', 'log');  % Set y-axis to log scale
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
ylabel('Channel Est. MSE (SNR=15dB)');
box on;
grid on;


%% Full-Atten arch
en1_dm4_da2_offline = [0.05272422358393669	0.01464245840907097	0.004593867342919111	0.001948320772498846	0.001448263297788799];
full_atten_dm1_da1_offline=[0.04411	0.01256	0.00457	0.00204	0.00152];
full_atten_dm2_da2_offline=[0.04608	0.01414	0.00517	0.00227	0.00166];
full_atten_dm4_da2_offline = [0.21903	0.20895	0.20668	0.20800	0.20836];

en1_dm4_da2_online = [0.05638652667403221	0.01563427411019802	0.00568010238930583	0.003263447666540742	0.002564249560236931];
full_atten_dm1_da1_online=[0.0479	0.0198	0.0128	0.0108	0.0103];
full_atten_dm2_da2_online=[0.0502	0.0210	0.0132	0.0109	0.0104];
full_atten_dm4_da2_online = [0.2037	0.1928	0.1950	0.1949	0.1942];

snr_index=4; % @15dB
offline_vals = [en1_dm4_da2_offline(snr_index), full_atten_dm1_da1_offline(snr_index), full_atten_dm2_da2_offline(snr_index), full_atten_dm4_da2_offline(snr_index)];
online_vals  = [en1_dm4_da2_online(snr_index),  full_atten_dm1_da1_online(snr_index),  full_atten_dm2_da2_online(snr_index),  full_atten_dm4_da2_online(snr_index)];
% Plot grouped bar chart
figure;
bar([offline_vals; online_vals]);   
% Set bar colors
loc_colors=[red;light_dark;bl;gr];
h = gca;
for i = 1:4 
    h.Children(5-i).FaceColor = loc_colors(i,:);
    h.Children(5-i).LineWidth = 1.5;  % Set border width
end

ylim([0.001 1])
set(gca, 'XTick', 1:2, 'XTickLabel', {'Offline-Pretrained','Online-Adapted'});
legend('$N_\mathrm{e}=1,N_\mathrm{dm}=4,N_\mathrm{ds}=2$','E2E trans.,$N_\mathrm{e}=1,L_\mathrm{dm}=1,L_\mathrm{ds}=1$',...
      'E2E trans.,$N_\mathrm{e}=1,L_\mathrm{dm}=2,L_\mathrm{ds}=2$',...
      'E2E trans.,$N_\mathrm{e}=1,L_\mathrm{dm}=4,L_\mathrm{ds}=2$', ...
      'Interpreter', 'latex', ...
      'FontSize',16);
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gca, 'YScale', 'log');  % Set y-axis to log scale
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
ylabel('Channel Est. MSE (SNR=15dB)');
box on;
grid on;

%% 
% figure;
% concat_off = [0.05166814476251602 0.01430347841233015 0.004764246754348278 0.00209481967613101 0.001428111456334591];
% ratio_off = [0.04994552582502365 0.01353888679295778 0.00445245997980237 0.002038480015471578 0.001489263027906418];

% % ratio_online=[0.056386527 0.015634274 0.005680102 0.003263448 0.00256425];
% x=[0 5 10 15 20];
% y=[concat_off',ratio_off'];
% loc_colors=[bl;red];
% loc_markers={'>','o'};
% snr=[0 5 10 15 20];
% for i = 1:size(y, 2)
%     semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',loc_markers{i});
%     hold on;
% end
% ylim([0.001 0.1]);
% xlabel("SNR (dB)")
% ylabel("Channel Est. MSE");
% set(gca,'Fontname','times new Roman','FontSize',24,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
% set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
% grid on;
% box on;
% legend('Concat-based input fusion','Ratio-based input fusion')


%% Input fusion: Concat or Ratio
figure;
models = {'Ratio','Concat'};
flops = [208224, 375840]./1e6;  % Total MFLOPs in the encoder, comparing stack_x and divise_x two methods
mse = [0.002038480015471578,0.00209481967613101];  % MSE at 20dB

% % Create scatter plot
% scatter(mse, flops_mflops, 150, 'filled', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);

% Set different colors for each model
loc_colors=[red;bl;or;gr];
loc_size = 300;
scatter(flops(1), mse(1), loc_size, loc_colors(1,:), 'filled', 's', ...
        'MarkerEdgeColor', 'black', 'LineWidth', 1.5);  % Square
hold on;
scatter(flops(2), mse(2), loc_size, loc_colors(2,:), 'filled', '^', ...
        'MarkerEdgeColor', 'black', 'LineWidth', 1.5);  % Triangle

ylim([0.0015 0.003]);
xlim([0.2 0.4]);
% Add grid for better readability
ylabel("Channel Est. MSE")
xlabel("Number of MFLOPs (Encoder)");
set(gca, 'YScale', 'log');
set(gca,'Fontname','times new Roman','FontSize',26,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
grid on;
box on;


%% Token: Symbol-wise vs Channel-wise
% Keep the same token pattern for both tasks
figure;
symbol_token_offline=[0.06208012998104095	0.01519526168704033	0.004279993940144777	0.001819362514652312	0.001332362997345626];
channel_token_offline=[0.05272422358393669	0.01464245840907097	0.004593867342919111	0.001948320772498846	0.001448263297788799];

symbol_token_pre_online=[0.06426767259836197	0.01850957237184048	0.00982699915766716	0.009105388075113297	0.009009788744151592];
channel_token_pre_online=[0.05846426263451576	0.01844874955713749	0.007442028261721134	0.00536695821210742	0.00492539070546627];

symbol_token_online=[0.05903296172618866	0.01729314029216766	0.009191406890749931	0.008769838139414787	0.008848275057971478];
channel_token_online=[0.05638652667403221	0.01563427411019802	0.00568010238930583	0.003263447666540742	0.002564249560236931];

x=[0 5 10 15 20];
y=[symbol_token_online',channel_token_online'];
loc_colors=[bl;red];
loc_markers={'>','o'};
snr=[0 5 10 15 20];
for i = 1:size(y, 2)
    semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',loc_markers{i});
    hold on;
end

ylim([0.001 0.1]);
xlabel("SNR (dB)")
ylabel("Channel Est. MSE");
set(gca,'Fontname','times new Roman','FontSize',26,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
grid on;
box on;
legend('Symbol-wise token','Channel-wise token')



%% Mask: Random-Symbol vs Random-RE
figure;

RanRE_off=[0.052417167 0.013861265 0.004752657 0.002198695 0.001570175];
RanSym_off=[0.052724224 0.014642458 0.004593867 0.001948321 0.001448263];
RanRE_pre_online=[0.058830924 0.017827654 0.007506567 0.005352434 0.004877591];
RanSym_pre_online=[0.058464263 0.01844875 0.007442028 0.005366958 0.004925391];
RanRE_online=[0.197071478 0.171757102 0.165456831 0.167091191 0.165640891];
RanSym_online=[0.056386527 0.015634274 0.005680102 0.003263448 0.00256425];

y=[RanRE_online',RanSym_online'];
loc_colors=[bl;red];
loc_markers={'>','o'};
snr=[0 5 10 15 20];
for i = 1:size(y, 2)
    semilogy(x, y(:, i), 'Color', loc_colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',loc_markers{i});
    hold on;
end

ylim([0.001 1]);
xlabel("SNR (dB)")
ylabel("Channel Est. MSE");
set(gca,'Fontname','times new Roman','FontSize',26,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
grid on;
box on;
legend('Random-RE masking','Random-symbol masking')


%% Online Transmit-Symbol Recovery
wo_ch_coding=[0.05638652667403221	0.01563427411019802	0.00568010238930583	0.003263447666540742	0.002564249560236931];
with_ch_coding=[0.05654119700193405	0.01597542129456997	0.005905142519623041	0.003402106696739793	0.002742309588938951];

figure;
snr = [0 5 10 15 20];
Y = [wo_ch_coding', with_ch_coding'];
b = bar(snr, Y);
loc_colors=[bl;or];
h = gca;
for i = 1:2
    h.Children(3-i).FaceColor = loc_colors(i,:);
    h.Children(3-i).LineWidth = 1.5;  % Set border width
end
xlabel("SNR (dB)");
ylabel("Channel Est. MSE");
ylim([0.001 0.1]);
legend("Channel-coded loop", "Uncoded loop",'FontSize',20);
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
set(gca, 'YScale', 'log');
grid on;
box on;

% analyze time consumption
time_wo_ch=[];
time_with_ch=[];

% analyze SER
avg_ser_wo_ch=0.0241297353059053;
avg_ser_with_ch=0.00298560014925897;
data = readtable('per_batch_ser.csv','VariableNamingRule', 'preserve');
std_ser_with_ch=std(data{:,2}); % 0.0071
std_ser_wo_ch=std(data{:,3}); % 0.0037
avg1=mean(data{:,2}); %???
avg2=mean(data{:,3}); %???


%% Flop comparison:
% Total Recovery FLOPs: 888,974,017
% Total Recovery FLOPs: 353,032,694
% Per-OFDM-frame recovery flops
figure;
flops_sym= 353032694/10000;
flops_bit= 888974017/10000;
y = [flops_sym,flops_bit];

h_bar = barh(y);  % Use barh for horizontal bars
h_bar.LineWidth = 1.5;  % Set bar border width to 1.5

% tell it to color each bar individually:
h_bar.FaceColor = 'flat';

% assign [R G B] to each bar (row 1 = first bar, row 2 = second bar)
h_bar.CData(1,:) = or;   % teal‑ish
h_bar.CData(2,:) =bl;   % orange‑ish

set(gca, 'YTick', 1:2, 'YTickLabel', {'Uncoded', 'Channel-coded'});  % Changed to YTick and YTickLabel

xlabel("Number of FLOPs");  % Changed to xlabel since it's now horizontal
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
ax = gca;
ax.YAxis.TickLabelRotation = 0;
grid on;
box on;


%% Online: SNR variation (line graph)
% Adapt-SNR vs Eval-SNR table converted to vectors
% Each vector represents evaluation at a specific SNR (0, 5, 10, 15, 20 dB)
% Rows correspond to adaptation SNR: 0, 5, 10, 15, 20 dB
figure;
eval_snr_0dB = [0.04169300198554993	0.02319975756108761	0.01886888034641743	0.01765635050833225	0.01738649420440197];
eval_snr_5dB = [0.0384615920484066	0.01301218848675489	0.006761365104466677	0.004978399258106947	0.004475715104490519];
eval_snr_10dB = [0.05185908824205399	0.01514122448861599	0.005940566305071115	0.003622700925916433	0.002898987149819732];
eval_snr_15dB = [0.06272026896476746	0.01779787614941597	0.006071957293897867	0.003464449662715197	0.002600667998194695];
eval_snr_20dB = [0.0691046416759491	0.01970198936760426	0.00632275827229023	0.003305039368569851	0.002335335128009319];
wo_adapt= [0.0584643  0.0184487  0.0074420 0.0053670 0.0049254];

y= [eval_snr_0dB', eval_snr_5dB', eval_snr_10dB', eval_snr_15dB', eval_snr_20dB', wo_adapt'];
x = [0 5 10 15 20];  % Adaptation SNR values
for i = 1:size(y, 2)
    semilogy(x, y(:, i), 'Color', colors(i, :), 'LineWidth', 2.5, 'MarkerSize',mark_size,'Marker',markers{i});
    hold on;
end

ylim([0.002 0.1]);
yticks([0.002 0.01 0.1]);
% ylim([0.001 0.5]);
% set(gca, 'YTickLabelMode', 'auto');
% ax = gca;
% ax.YAxis.Exponent = 0;  % Force scientific notation

xlabel("SNR (dB)")
ylabel("Channel Est. MSE");
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
grid on;
box on;
legend('Adapt. at 0dB', ...
        'Adapt. at 5dB', ...
        'Adapt. at 10dB',...
        'Adapt. at 15dB',...
        'Adapt. at 20dB',...
        'Without adapt.', ...
        'Location', 'best','FontSize',18);



%% Online Data aug.
figure;
aug1 = [0.05804722011089325 0.01644443720579147 0.006124963518232107 0.003642425406724215 0.003233056981116533];
aug3 = [0.05619017407298088 0.01624259725213051 0.005927936639636755 0.003432044526562095 0.002734546316787601];
aug5 = [0.05638652667403221 0.01563427411019802 0.00568010238930583 0.003263447666540742 0.002564249560236931];
aug7 = [0.05515530705451965 0.01567905396223068 0.005627068690955639 0.003283469937741756 0.002759968861937523];

% Extract SNR 0, 10, 20 dB (indices 1, 3, 5) and calculate gains
snr_idx = [1,2,3,4,5];
augs = [aug3; aug5; aug7];  % Each row is one augmentation level
y_data = 10*log10(aug1(snr_idx) ./ augs(:, snr_idx));  % 3x3 matrix: rows=aug levels, cols=SNR

% Create grouped bar plot (transpose so SNR is on x-axis)
h_bar = bar(y_data');
loc_colors=[bl;or;gr];
for i = 1:3
    h_bar(i).LineWidth = 1.5;
    h_bar(i).FaceColor = loc_colors(i,:);
end

set(gca, 'XTick', 1:5, 'XTickLabel', {'0','5','10','15','20'});
ylim([0 1]);
xlabel("SNR (dB)");
ylabel("MSE Gain (dB) over A=1");
legend('Aug=3', 'Aug=5', 'Aug=7', 'Location', 'best');
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 16 14],'DefaultLineLineWidth',2.5);
grid on; box on;


%% E2E vs encoder only
figure;
snr=[0 5 10 15 20];
pre_adapt = [0.0584643, 0.0184487, 0.0074420, 0.0053670, 0.0049254];
e2e_adapt = [0.0629285, 0.0175458, 0.0063313, 0.0038047, 0.0031366];
enc_only_adapt = [0.0563865, 0.0156343, 0.0056801, 0.0032634, 0.0025642];


e2e_gain = 10*log10(e2e_adapt./pre_adapt);
enc_only_gain = 10*log10(enc_only_adapt./pre_adapt);

% bar plot
Y = [-e2e_gain', -enc_only_gain'];
b = bar(snr, Y);

loc_colors=[bl;or];
h = gca;
for i = 1:2
    h.Children(3-i).FaceColor = loc_colors(i,:);
    h.Children(3-i).LineWidth = 1.5;  % Set border width
end

xlabel("SNR (dB)");
ylabel("MSE Gain (dB)");
legend("Full-SSL-branch update", "Encoder-only update",'FontSize',20);
set(gca,'Fontname','times new Roman','FontSize',22,'Linewidth',1.5,'GridAlpha',.5,'GridLineStyle',':');
set(gcf,'unit', 'centimeters', 'position', [10 5 18 12],'DefaultLineLineWidth',2.5);
grid on;
box on;
