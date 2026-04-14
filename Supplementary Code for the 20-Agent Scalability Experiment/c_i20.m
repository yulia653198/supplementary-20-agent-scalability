clear; close all; clc;

%% 0) Parameters
T    = 8;
dt   = 0.01;

N = 20;         % followers
n = 2;          % 2D state
baseSeed = 12345;

%% --- Plot palette ---
co    = turbo(N);   % stronger color separation than lines(N)
label = arrayfun(@(i) sprintf('Follower %d', i), 1:N, 'uni', 0);

%% 1) System
A = [ 0   1.5;
     -2   0 ];
B = [0; 2];

Q = eye(n); 
R = 1;
[P,~,~] = care(A,B,Q,R);
K = (B') * P;
disp('K ='); 
disp(K);

Ad = expm(A*dt);     % P1 predictor (A0 = A => same Ad)

%% 2) Fixed topology + pinning (20-agent version)
% Directed chain + extra forward links
A_f = zeros(N,N);

% backbone: 1->2->3->...->20
for i = 2:N
    A_f(i, i-1) = 1;
end

% extra directed links: j=i-2 -> i
for i = 3:N
    A_f(i, i-2) = 1;
end

% a few longer-range links
for i = 5:5:N
    A_f(i,1) = 1;
end
for i = 8:4:N
    A_f(i, max(1,i-4)) = 1;
end

% all followers pinned to leader
g = ones(N,1);

a_bar = A_f;
g_bar = g;

%% 3) Event-trigger params (followers)
c1_et      = 0.5;
alpha_et   = 0.30;
hmin_et    = 0.05;
hthr_floor = 0.03;

rng(baseSeed+999, 'twister');
rho_et = 0.85 + 0.30*rand(N,1);

%% 4) Attack params (Multi-type)
% FF links: follower -> follower
delta_FF   = 0.10;
sigx_FF0   = 0.12;
sigu_FF0   = 0.05;

% LF links: leader -> follower
delta_LF   = 0.08;
sigx_LF0   = 0.10;
sigu_LF0   = 0.04;

atk_decay  = 0.8;
atk_floor  = 0.00;

% Residual gate thresholds
gamma_FF = 0.65;
gamma_LF = 0.65;

%% 5) Leakage-modified adaptive-law parameters
% \dot{\hat c}_i = ||zeta_i||^2 / \hat c_i^2 - sigma_c * (\hat c_i - c0)
c0      = 1.0;    % reliability-oriented baseline level
sigma_c = 0.35;   % leakage coefficient

%% 6) Simulation init
time = 0:dt:T;
numSteps = numel(time);

rng(baseSeed+1, 'twister');
x  = rand(n,N);
x0 = [2;2];       % leader state

% adaptive gain initialization: must satisfy c_i(0) >= c0
c  = c0 * ones(N,1);
c_store = zeros(N, numSteps);

% network buffers include both follower-follower estimates and leader estimates
net = init_network_buffers(x, x0, n, N);

% initialize held zeta at t=0
zeta0 = compute_zeta(x, net.xhat0, net.xhat, a_bar, g_bar);
net.zeta_hold = zeta0;
net.t_tx(:) = 0;

%% 7) Main loop
eps_den = 1e-12;

for k = 1:numSteps
    t = time(k);

    % (1) P1 predictor step for ALL estimated signals (FF + LF)
    net = predictor_step_P1(net, Ad);

    % (2) Leader broadcast update (LF deception + residual gate + Strategy C)
    [sigx_LF, sigu_LF] = attack_sigma(sigx_LF0, sigu_LF0, atk_decay, atk_floor, t);
    net = leader_broadcast_update(net, x0, t, delta_LF, sigx_LF, sigu_LF, gamma_LF);

    % (3) current zeta
    zeta = compute_zeta(x, net.xhat0, net.xhat, a_bar, g_bar);

    % (4) Follower ET update + broadcast (FF deception + residual gate + Strategy C)
    [sigx_FF, sigu_FF] = attack_sigma(sigx_FF0, sigu_FF0, atk_decay, atk_floor, t);
    net = event_trigger_update_hold_and_broadcast(net, x, zeta, a_bar, t, ...
        c1_et, alpha_et, hmin_et, hthr_floor, rho_et, ...
        delta_FF, sigx_FF, sigu_FF, gamma_FF);

    % (5) Leakage-modified adaptive law uses held zeta
    for i = 1:N
        chat_i = c(i);
        zi = net.zeta_hold(:,i);

        cdot = (zi' * zi) / (chat_i^2 + eps_den) ...
             - sigma_c * (chat_i - c0);

        c_new = chat_i + dt * cdot;

        % numerical safeguard to keep c_i(t) >= c0
        c(i) = max(c0, c_new);
    end

    % (6) Control uses held zeta
    u = zeros(N,1);
    for i = 1:N
        u(i) = -c(i) * (K * net.zeta_hold(:,i));
    end

    % (7) Dynamics
    for i = 1:N
        x(:,i) = x(:,i) + dt * (A*x(:,i) + B*u(i));
    end

    % leader dynamics
    x0 = x0 + dt * (A*x0);

    % (8) store c
    c_store(:,k) = c;
end

%% 8) Plot: c_i(t)
figure('Name','$\hat c_i(t)$', 'Color', 'w');
hold on; grid on; box on;

ax = gca;
set(ax, 'FontName', 'Times New Roman', ...
        'FontSize', 14, ...
        'LineWidth', 1.0, ...
        'TickDir', 'in');

% line styles cycle
ls = {'-','--',':','-.'};

h = gobjects(N,1);
for i = 1:N
    h(i) = plot(time, c_store(i,:), ...
                'LineWidth', 1.3, ...
                'Color', co(i,:), ...
                'LineStyle', ls{mod(i-1, numel(ls))+1}, ...
                'DisplayName', label{i});
end

xlabel('Time (s)', 'Interpreter', 'latex');
ylabel('$\hat c_i(t)$', 'Interpreter', 'latex');

yl = ylim;
padY = 0.20 * max(1e-9, yl(2) - yl(1));
ylim([yl(1)-padY, yl(2)+padY]);

lg = legend(h, label, ...
            'Location', 'eastoutside', ...
            'Interpreter', 'latex', ...
            'Box', 'on');
if isprop(lg, 'NumColumns')
    lg.NumColumns = 2;
end
lg.FontSize  = 14;
lg.LineWidth = 1.0;

%% ========================================================================
%  Local functions
%% ========================================================================

function net = init_network_buffers(x_init, x0_init, n, N)
    net.t_tx = -1e9 * ones(N,1);
    net.zeta_hold = zeros(n,N);

    % xhat(:,i,j): receiver i's effective signal for link j->i (FF links)
    net.xhat = zeros(n,N,N);
    for i = 1:N
        for j = 1:N
            net.xhat(:,i,j) = x_init(:,j);
        end
    end

    % xhat0(:,i): receiver i's effective signal for leader link 0->i (LF links)
    net.xhat0 = zeros(n,N);
    for i = 1:N
        net.xhat0(:,i) = x0_init;
    end
end

function net = predictor_step_P1(net, Ad)
    [~,N,~] = size(net.xhat);
    for i = 1:N
        for j = 1:N
            net.xhat(:,i,j) = Ad * net.xhat(:,i,j);
        end
        net.xhat0(:,i) = Ad * net.xhat0(:,i);
    end
end

function zeta = compute_zeta(x, xhat0, xhat, a_bar, g_bar)
    [n,N] = size(x);
    zeta = zeros(n,N);
    for i = 1:N
        ssum = zeros(n,1);
        for j = 1:N
            if a_bar(i,j) ~= 0
                ssum = ssum + a_bar(i,j) * (x(:,i) - xhat(:,i,j));
            end
        end
        zeta(:,i) = ssum + g_bar(i) * (x(:,i) - xhat0(:,i));
    end
end

function [sigx, sigu] = attack_sigma(atkSigma_x0, atkSigma_u0, atk_decay, atk_floor, t)
    sigx = max(atk_floor, atkSigma_x0 * exp(-atk_decay*t));
    sigu = max(atk_floor, atkSigma_u0 * exp(-atk_decay*t));
end

function net = leader_broadcast_update(net, x0, t, delta_LF, sigx, sigu, gamma_LF) %#ok<INUSD>
    [n,N] = size(net.xhat0);
    x_tx = x0;

    for i = 1:N
        if rand < delta_LF
            x_recv = x_tx + sigx * randn(n,1);
            u_recv = sigu * randn(1,1); %#ok<NASGU>
        else
            x_recv = x_tx;
            u_recv = 0; %#ok<NASGU>
        end

        r_i0 = norm(x_recv - net.xhat0(:,i));
        s_i0 = (r_i0 <= gamma_LF);

        if s_i0
            net.xhat0(:,i) = x_recv;
        end
    end
end

function net = event_trigger_update_hold_and_broadcast(net, x, zeta, a_bar, t, ...
    c1_et, alpha_et, hmin_et, hthr_floor, rho_et, ...
    delta_FF, sigx, sigu, gamma_FF) %#ok<INUSD>

    [n,N] = size(x);

    for j = 1:N
        % Trigger on held-zeta error (sender-side)
        e_j   = net.zeta_hold(:,j) - zeta(:,j);
        thr_j = max(hthr_floor, rho_et(j) * c1_et * exp(-alpha_et*t));
        trig_j = (norm(e_j) >= thr_j) && ((t - net.t_tx(j)) >= hmin_et);

        if trig_j
            % update held zeta at sender j
            net.zeta_hold(:,j) = zeta(:,j);
            net.t_tx(j) = t;

            % broadcast x_j
            x_tx = x(:,j);

            for i = 1:N
                if a_bar(i,j) ~= 0
                    % FF deception on link j->i
                    if rand < delta_FF
                        x_recv = x_tx + sigx * randn(n,1);
                        u_recv = sigu * randn(1,1); %#ok<NASGU>
                    else
                        x_recv = x_tx;
                        u_recv = 0; %#ok<NASGU>
                    end

                    % residual gate + Strategy C
                    r_ij = norm(x_recv - net.xhat(:,i,j));
                    s_ij = (r_ij <= gamma_FF);

                    if s_ij
                        net.xhat(:,i,j) = x_recv;
                    end
                end
            end
        end
    end
end