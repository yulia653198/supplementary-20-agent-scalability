%% 多智能体系统稳定性仿真（20-agent version）
% 只保留图2：Tracking errors of x1/x2/x3
clear; close all; clc;

%% 0) 全局参数
T    = 15;                 % 仿真总时长 [s]
dt   = 0.01;               % 步长 [s]
time = 0:dt:T;
numSteps = numel(time);

N  = 20;                   % 跟随者数（改为20）
n  = 3;                    % 状态维数

MC = 100;                  % Monte Carlo 次数
baseSeed = 12345;          % 共同随机数基种子

%% 1) 系统矩阵（直升机样）& LQR 增益 K = B'*P
A = [-0.8,   0,    -0.04;
      2.0,   0,     0.00;
     -1.8,  6.8,   -0.06];
B = [3.3; 0; 6.8];
Q = eye(3); 
R = 1;
[P,~,~] = care(A, B, Q, R);
K = (B') * P;              % 1×3
disp('K =');
disp(K);

%% 2) 通信拓扑（20-agent directed）与钉扎（pinning）
% A_f(i,j)=1 表示 j -> i 有向边（i接收j）
A_f = zeros(N,N);

% 主链路：1->2->3->...->20
for i = 2:N
    A_f(i, i-1) = 1;
end

% 次链路：i-2 -> i
for i = 3:N
    A_f(i, i-2) = 1;
end

% 少量长程边
for i = 5:5:N
    A_f(i,1) = 1;
end
for i = 8:4:N
    A_f(i, max(1,i-4)) = 1;
end

% pinning：前 6 个 follower 接 leader
g = zeros(N,1); 
g(1:6) = 1;

%% 3) 双伯努利激活概率（边 b_ij 与 钉扎 beta_i）
pA = 0.85 * ones(N,N);     
pA(~logical(A_f)) = 0;

pg = 0.90 * ones(N,1);     
pg(~logical(g)) = 0;

%% 4) 可靠性常数（耦合基线）
c0 = 0.5;                  
eps_den = 1e-12;           

%% 5) 单次自适应仿真（仅为绘制图2保存轨迹）
rng(baseSeed+1,'twister');

x   = rand(n,N);            
x0s = [0.1;0;0];
c_s = ones(N,1);

x_store  = zeros(n,N,numSteps);
x0_store = zeros(n,numSteps);

for k = 1:numSteps
    t = time(k);

    % --- 双伯努利 + 确定性扰动 ---
    a_bar = zeros(N,N);
    if any(A_f(:))
        bij  = double(rand(N,N) < pA);
        detA = (1 + 0.5 * sin(t) .* rand(N,N));
        a_bar = (A_f .* detA) .* bij;
    end

    g_bar = zeros(N,1);
    if any(g)
        beta = double(rand(N,1) < pg);
        detg = (1 + 0.5 * sin(t) .* rand(N,1));
        g_bar = (g .* detg) .* beta;
    end

    % --- zeta ---
    zeta = zeros(n,N);
    for i = 1:N
        ssum = zeros(n,1);
        for j = 1:N
            if a_bar(i,j) ~= 0
                ssum = ssum + a_bar(i,j) * (x(:,i) - x(:,j));
            end
        end
        zeta(:,i) = ssum + g_bar(i) * (x(:,i) - x0s);
    end

    % --- 自适应律 + 控制 ---
    c_dot = zeros(N,1);
    for i = 1:N
        chat_i = c0 + c_s(i);
        c_dot(i) = (zeta(:,i).' * zeta(:,i)) / (chat_i^2 + eps_den);
    end
    c_s = c_s + dt*c_dot;

    u_s = zeros(N,1);
    for i = 1:N
        u_s(i) = - (c0 + c_s(i)) * (K * zeta(:,i));
    end

    % --- 更新 & 存储 ---
    for i = 1:N
        x(:,i) = x(:,i) + dt*(A*x(:,i) + B*u_s(i));
    end
    x0s = x0s + dt*(A*x0s);

    x_store(:,:,k) = x;
    x0_store(:,k)  = x0s;
end

%% 6) 只画图2：Tracking error vs Time — 三个子图 (x1, x2, x3)
Tz  = min(15, T);                 
idz = time <= Tz;
tt  = time(idz);

figure('Name','Tracking errors of x1/x2/x3','Color','w','Position',[80 100 1200 760]);
tl = tiledlayout(3,1,'Padding','compact','TileSpacing','loose');

co = turbo(N);   % 20个智能体颜色更容易区分
ls = {'-','--',':','-.'};

for comp = 1:3
    ax = nexttile(tl); 
    hold(ax,'on'); 
    grid(ax,'on'); 
    box(ax,'on');

    % 计算所有跟随者在该分量的 tracking error: x_i(comp) - x0(comp)
    for i = 1:N
        ei = squeeze(x_store(comp,i,idz))' - x0_store(comp, idz);  
        plot(ax, tt, ei, ...
             'LineWidth', 1.6, ...
             'LineStyle', ls{mod(i-1,numel(ls))+1}, ...
             'Color', co(i,:));
    end

    set(ax,'FontName','Times New Roman','FontSize',16, ...
           'LineWidth',1.0,'TickDir','in');
    xlim(ax,[0 Tz]);

    % 自动加一点上下边距
    yl = ylim(ax);
    dy = 0.08 * max(1e-6, yl(2)-yl(1));
    ylim(ax, [yl(1)-dy, yl(2)+dy]);

    ylabel(ax, sprintf('$x_{%d}$', comp),'Interpreter','latex');

    % 只在第一个子图放图例
    if comp==1
        lg = legend(ax, arrayfun(@(i) sprintf('Follower %d', i), 1:N, 'uni',0), ...
                    'Location','eastoutside', ...
                    'Box','on');
        if isprop(lg,'NumColumns')
            lg.NumColumns = 2;
        end
        lg.FontSize = 14;
        lg.LineWidth = 1.0;
    end
end

xlabel(nexttile(tl,3),'Time (s)','Interpreter','latex');