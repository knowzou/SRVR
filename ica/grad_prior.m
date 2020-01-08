function grad = grad_prior(w, flag)
if flag==1
    grad = w/(0.1*norm(w)^2 + 1);
else
    grad = w;
end
    
end