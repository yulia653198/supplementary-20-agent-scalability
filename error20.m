
clear; close all; clc;


T    = 15;                 
dt   = 0.01;              
time = 0:dt:T;
numSteps = numel(time);

N  = 20;                  
n  = 3;                   

MC = 100;                  
baseSeed = 12345;          


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


A_f = zeros(N,N);


for i = 2:N
    A_f(i, i-1) = 1;
end

% 次链路：i-2 -> i
for i = 3:N
    A_f(i, i-2) = 1;
end


for i = 5:5:N
    A_f(i,1) = 1;
end
for i = 8:4:N
    A_f(i, max(1,i-4)) = 1;
end

g = zeros(N,1); 
g(1:6) = 1;


pA = 0.85 * ones(N,N);     
pA(~logical(A_f)) = 0;

pg = 0.90 * ones(N,1);     
pg(~logical(g)) = 0;


c0 = 0.5;                  
eps_den = 1e-12;           

rng(baseSeed+1,'twister');

x   = rand(n,N);            
x0s = [0.1;0;0];
c_s = ones(N,1);

x_store  = zeros(n,N,numSteps);
x0_store = zeros(n,numSteps);

for k = 1:numSteps
    t = time(k);


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

 
    for i = 1:N
        x(:,i) = x(:,i) + dt*(A*x(:,i) + B*u_s(i));
    end
    x0s = x0s + dt*(A*x0s);

    x_store(:,:,k) = x;
    x0_store(:,k)  = x0s;
end


Tz  = min(15, T);                 
idz = time <= Tz;
tt  = time(idz);

figure('Name','Tracking errors of x1/x2/x3','Color','w','Position',[80 100 1200 760]);
tl = tiledlayout(3,1,'Padding','compact','TileSpacing','loose');

co = turbo(N);  
ls = {'-','--',':','-.'};

for comp = 1:3
    ax = nexttile(tl); 
    hold(ax,'on'); 
    grid(ax,'on'); 
    box(ax,'on');

 
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


    yl = ylim(ax);
    dy = 0.08 * max(1e-6, yl(2)-yl(1));
    ylim(ax, [yl(1)-dy, yl(2)+dy]);

    ylabel(ax, sprintf('$x_{%d}$', comp),'Interpreter','latex');


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
