function grad = grad_func(W, X)
[row, col] = size(X);
grad = zeros(row, row);
for i = 1: col
    x = X(:,i);
    grad = grad  - (tanh(1/2*W'*x)*x')' + (W')^(-1);
end
grad = -grad/col;
end