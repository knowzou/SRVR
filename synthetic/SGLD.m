function [datapass, W_record] = SGLD(X, param, obj)
[d, n] = size(X);


%parameters
w = param.w0;
eta = param.eta;
epochNum = param.epochNum;
batchSize = param.batchSize;
W_record = zeros(d, epochNum);
datapass = zeros(1, epochNum);

gradNum = 0;
%w_record = zeros(d, epochNum/record_step); 
for j = 1:epochNum

    
    items = randsample(n, batchSize, false);

    w = w - grad_func(w, X(:,items), obj)*eta+ sqrt(2*eta)*randn(d,1);
    gradNum = gradNum + batchSize;
    
    %record
    W_record(:,j) = w;
    datapass(j) = gradNum/n;
    
end
end

