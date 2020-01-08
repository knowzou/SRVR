function [datapass, W_record] = SGHMC(X, param, obj)
[d, n] = size(X);


%parameters
w = param.w0;
eta = param.eta;
epochNum = param.epochNum;
batchSize = param.batchSize;
W_record = zeros(d,epochNum);
datapass = zeros(1,epochNum);
gamma = param.gamma;
u = param.u;

w_v = zeros(d,1);
gradNum = 0;

%w_record = zeros(d, epochNum/record_step);
for j = 1:epochNum
    
    %stochastic gradient
    items = randsample(n, batchSize, false);
    grad = grad_func(w, X(:,items), obj);
    gradNum = gradNum + batchSize;
    %update
    [xi_v, xi_x] = brown(u,gamma,eta,d);
    w_v_temp = w_v*(1-gamma*eta) - u*eta*grad + sqrt(2*gamma*u*eta)*randn(d,1);
    w = w +eta*w_v;
    w_v = w_v_temp;
    
    
    
    %record iterates
    W_record(:,j) = w;
    datapass(j) = gradNum/n;
end
end

