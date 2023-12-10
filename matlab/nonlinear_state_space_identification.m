pickle = py.importlib.import_module('pickle');
fh = py.open('../data/test_set_prbs.pkl', 'rb')
P = pickle.load(fh);    % pickle file loaded to Python variable
fh.close();
mP = py2mat(P); 
%%
u = squeeze(mP.u_test(1,:,:));
y = mP.y_test(1,:,2)';
z = iddata(y,u);

plot(z)

%%
% il mio modello è x_{k+1} = a * x_1 + b_x2
