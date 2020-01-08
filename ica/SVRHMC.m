function [beta_record, datapass, elapse] = SVRHMC(X, param)
% the class y should be +1, -1

% parameter
u = param.u;
gamma = param.gamma;
eta = param.eta;
batch_size = param.batchSize;
epoch_num = param.epochNum;
burn_in = param.burnIn;
lambda = param.lambda;
LbatchSize  = param.LbatchSize;

% data and dataset parameter
[d, N] = size(X);

%initialization
beta = param.x0;
beta_v = zeros(d,1);
%beta_v = randn(size(X,1),1);%momentum
beta_burn = zeros(d,d); % path average
loglike = zeros(1,epoch_num);
error = zeros(1,epoch_num);
datapass = zeros(1,epoch_num+1);
elapse = zeros(1,epoch_num);
beta_record = [];

count = 0;
t = 1;
gradNum = 0;
%vrSGLD
disp('ASVRHMC')
tic
for j = 1:epoch_num

    beta_pre = beta;
    ind = randsample(N,LbatchSize,false);
    grad_pre = grad_func(beta, X(:,ind))*N;
    gradNum = gradNum + LbatchSize;
    grad = grad_pre;
    
    beta = beta +(1-exp(-gamma*eta))/gamma*beta_v ...
        + u*(gamma*eta+exp(-gamma*eta)-1)/gamma^2*grad;
    beta_v = beta_v*exp(-gamma*eta) - u*(1-exp(-gamma*eta))/gamma*grad ...
        + sqrt(u*(1 - exp(-2*gamma*eta)))*randn(d, d);
    
    for i=1:(floor(LbatchSize/batch_size))
        
        ind = randsample(N,batch_size,false);
        grad = (grad_func(beta, X(:,ind))...
            -grad_func(beta_pre, X(:,ind)))*N+grad_pre ;
%         grad_pre = grad;
        grad = grad + lambda*beta;
        
       % beta_pre = beta;
        beta = beta +(1-exp(-gamma*eta))/gamma*beta_v ...
            + u*(gamma*eta+exp(-gamma*eta)-1)/gamma^2*grad;
        beta_v = beta_v*exp(-gamma*eta) - u*(1-exp(-gamma*eta))/gamma*grad ...
            + sqrt(u*(1 - exp(-2*gamma*eta)))*randn(d, d);
        
        gradNum = gradNum + batch_size;
        count =(j-1)*floor(LbatchSize/batch_size)+i ;
        %eta = eta * count^(-0.001);
        if count>=burn_in
            beta_burn = beta_burn * (count-burn_in)/(count-burn_in+1)+ beta /(count-burn_in+1);
            if ~rem(i, 10)
                %                 loglike(t) = obj_func(beta_burn, X);
                beta_record = [beta_record, reshape(beta_burn,  d^2, 1)];
                elapse(t) = toc;
                datapass(t) = gradNum/N;
                t = t + 1;
            end
        else
            if ~rem(i, 10)
                %loglike(t) = obj_func(beta, X);
                elapse(t) = toc;
                datapass(t) = gradNum/N;
                beta_record = [beta_record, reshape(beta,  d^2, 1)];
                t = t + 1;
            end
        end
    end
end

end

