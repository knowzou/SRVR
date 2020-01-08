function grad = grad_func(w, X, obj)
[d, n] = size(X);
if strcmp(obj, 'normal')
    sigma = 2;
    grad = mean(w - X, 2)/sigma^2;
elseif  strcmp(obj, 'lognormal')
    grad = 0;
elseif strcmp(obj, 'binormal')
    grad = mean((repmat(w,1,n) - X)+ (2*X).*repmat((1+exp(2*w'*X)).^(-1),d,1),2);
elseif strcmp(obj, 'asy_binormal')
    grad = mean((repmat(w,1,n) - X)+ (2*X).*repmat((1+2*exp(2*w'*X)).^(-1),d,1),2);
end
end
