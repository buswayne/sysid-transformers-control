close all
clear all
clc

s = tf('s');
M = 1 / (3*s + 1);
W = 1 / (0.3*s + 1);
m1 = 1;
m2 = 0.5;
c1 = 0.2;
c2 = 0.5;
k1 = 1;
k2 = 0.5;

Ts = 0.1; Fs = 1/Ts;
T = 500;
t = 0:Ts:T-Ts;

T_u = 120; % s
u = zeros(size(t));
u((t >= 1*T_u/2) & (t < 2*T_u/2)) = 1;
u((t >= 3*T_u/2) & (t < 4*T_u/2)) = 1;
u((t >= 5*T_u/2) & (t < 6*T_u/2)) = 1;
u((t >= 7*T_u/2) & (t < 8*T_u/2)) = 1;
u = u';
%%

% u = 10*randn(size(t));

P = ( m1*s^2 + (c1+c2)*s + (k1+k2) ) / ( (m1*s^2 + (c1+c2)*s + k1+k2)*(m2*s^2+c2*s+k2) - (k2+c2*s)^2 );

y_0 = lsim(P, u, t);
y_n = sqrt(0.025)*randn(size(t))';
y = y_0 + y_n;

figure
plot(t, u)
hold on
plot(t, y)

%% VRFT
M = c2d(M,Ts);
M.Variable = 'z^-1';
W = c2d(W,Ts);
W.Variable = 'z^-1';
P = c2d(P,Ts);
P.Variable = 'z^-1';

%%
B=[tf([1],[1],Ts,'Variable','z^-1');
   tf(Ts*[1 1],2*[1 -1],Ts,'Variable','z^-1');
   tf([2 -2],Ts*[3 -1],Ts,'Variable','z^-1')];

[C, theta] = VRFT1_ry(u,y,M,B,W,4,[]);

theta

F = C*P/(1+C*P);

theta_paper = [0.0683, 0.1105, 0.2307];
C_paper = theta_paper*B;
F_paper = C_paper*P/(1+C_paper*P);


figure
step(F)
hold on
step(F_paper)

