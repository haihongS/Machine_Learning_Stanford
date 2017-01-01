function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

[row,col]=size(theta);
for i=1:m
	v=(theta')*(X(i,:)');
	v=sigmoid(v);
	J=J+y(i)*log(v)+(1-y(i))*log(1-v);
end
J=J/(-m);

for i=1:row
	for j=1:col
		for k=1:m
			v=sigmoid(theta'*(X(k,:)'));
			v=v-y(k);
			v=v*X(k,i);
			grad(i,j)=grad(i,j)+v;
		end
		grad(i,j)=grad(i,j)/m;
	end
end



% =============================================================

end