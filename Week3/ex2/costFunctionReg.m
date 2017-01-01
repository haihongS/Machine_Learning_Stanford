function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

for i=1:m
	v=sigmoid(theta'*X(i,:)');
	J=J+y(i)*log(v)+(1-y(i))*log(1-v);
end

J=J/(-m);

[row,col]=size(theta);

for i=2:row
	for j=1:col
		J=J+lambda/(2*m)*theta(i,j)*theta(i,j);
	end
end

for k=1:col
	for i=1:m
		v=sigmoid(theta'*X(i,:)');
		grad(1,k)=grad(1,k)+(v-y(i))*X(i,1);
	end
	grad(1,k)=grad(1,k)/m;
end

for i=2:row
	for j=1:col
		for k=1:m
			v=sigmoid(theta'*X(k,:)');
			grad(i,j)=grad(i,j)+(v-y(k))*X(k,i);
		end
		grad(i,j)=grad(i,j)/m;
		grad(i,j)=grad(i,j)+lambda/m*theta(i,j);
	end
end


% =============================================================

end
