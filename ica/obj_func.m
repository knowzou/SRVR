function obj = obj_func (W, X) 
% negative log likelihood
[row, col] = size(X);
obj = 0;
for i = 1:col
    x = X(:,i);
    obj = obj - sum(log(1./(4*(cosh(W'*x/2).^2)))) - log(abs(det(W)));
end
obj = obj/col;
end