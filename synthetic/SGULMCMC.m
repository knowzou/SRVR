function [datapass, W_record] = SGULMCMC(X, param, obj)
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
    w_v_temp = w_v*exp(-gamma*eta) - u*(1-exp(-gamma*eta))/gamma*grad + xi_v;%sqrt(2*gamma*u*eta)*randn(d,1);
    w = w +(1-exp(-gamma*eta))/gamma*w_v ...
        + u*(gamma*eta+exp(-gamma*eta)-1)/gamma^2*grad + xi_x;
    w_v = w_v_temp;
    
    
    
    %record iterates
    W_record(:,j) = w;
    datapass(j) = gradNum/n;
end
end

