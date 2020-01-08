function [loglike, datapass, elapse] = LMC(X, param)
% the class y should be +1, -1

% parameter
eta = param.eta;

epoch_num = param.epochNum;
lambda = param.lambda;


% data and dataset parameter
[d, N] = size(X);
batch_size = N;
%initialization

beta = param.x0;
beta_burn = zeros(d,d); % path average
%loglike = zeros(1,floor(epoch_num/N));
elapse = zeros(1,floor(epoch_num/N));

t = 1;
gradNum = 0;
%SGLD
disp('LMC')
tic

for j = 1:epoch_num
    eta = eta ;
    ind = randsample(N,batch_size,true);
    grad = grad_func(beta,X(:,ind))*N + lambda*beta;
    beta = beta - grad*eta;% + 1*sqrt(2*eta)*randn(d,d);
    gradNum = gradNum + batch_size;
    if ~rem(gradNum,100)
        disp(strcat('total dataPass:  ', num2str(epoch_num*batch_size/N),'  current dataPass:  ', num2str(gradNum/N)));
    end
    datapass(j) = gradNum/N;
    loglike(j) = obj_func(beta, X);
    
 
end


end

