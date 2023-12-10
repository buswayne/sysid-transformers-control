close all
clear all
clc

syms X2 P2 X2_dot P2_dot
syms P100 F200
syms M C UA2 Cp lambda lambda_s F1 X1 F3 T1 T200 

T2 = 0.5616*P2 + 0.3126*X2 + 48.43;
T3 = 0.507*P2 + 55;
T100 = 0.1538*P100 + 90;
UA1 = 0.16*(F1 + F3);
Q100 = UA1*(T100 - T2);
F100 = Q100/lambda_s;
Q200 = (UA2*(T3 - T200))/(1 + UA2/(2*Cp*F200));
F5 = Q200/lambda;
F4 = (Q100 - F1*Cp*(T2 - T1)) / lambda;
F2 = F1 - F4;


eq1 = M*X2_dot == F1*X1 - F2*X2;
eq2 = C*P2_dot == F4 - F5;

