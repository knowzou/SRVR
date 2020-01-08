function [datapass, w_record] = SVRHMC(X, param, obj)
[d, n] = size(X);


%parameters
w = param.w0;
eta = param.eta;
epochNum = param.epochNum;
minibatchSize = param.minibatchSize;
u = param.u;
gamma = param.gamma;
w_v = zeros(d,1);

len = floor(epochNum*n/minibatchSize);
w_record = zeros(d, len);
datapass = zeros(1, len);

gradNum = 0;
t = 0;
for j = 1:epochNum
    
    %compute full gradient
    grad_pre = grad_func(w, X, obj);
    w_pre = w;
    gradNum = gradNum + n;
    
    for k = 1: n/minibatchSize
        items = randsample(n, minibatchSize, true);
        
        %semi-stochastic gradient
        grad = (grad_func(w, X(:, items), obj) - grad_func(w_pre, X(:, items), obj))...
            + grad_pre;
        gradNum = gradNum + minibatchSize;
        
        %update
        [xi_v, xi_x] = brown(u,gamma,eta,d);
        w_v = w_v*exp(-gamma*eta) - u*(1-exp(-gamma*eta))/gamma*grad + xi_v;
        w = w +(1-exp(-gamma*eta))/gamma*w_v ...
            + u*(gamma*eta+exp(-gamma*eta)-1)/gamma^2*grad + xi_x;
        %w_v = w_v_temp;
        
        
        %record the iterate
        t = t+1;
        w_record(:,t) = w;
        datapass(t) = gradNum/n;
    end
    
    %record sample path
end
end

