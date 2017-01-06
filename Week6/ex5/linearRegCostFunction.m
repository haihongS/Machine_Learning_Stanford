function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h=zeros(m,1);
h=X*theta;
h=h.-y;
h=h.^2;
h=h./(2*m);
J=sum(h(:));

pp=theta.*theta;
pp(1,1)=0;
J=J+sum(pp(:))*lambda/(2*m);

len=size(theta,1);
h=X*theta;
h=h.-y;
tt=h.*X(:,1);
grad(1,1)=sum(tt(:));
grad(1,1)=grad(1,1)/m;

for j=2:len
	tt=h.*X(:,j);
	grad(j,1)=sum(tt(:))/m;
	grad(j,1)=grad(j,1)+lambda/m*theta(j,1);
end












% =========================================================================

grad = grad(:);

end
