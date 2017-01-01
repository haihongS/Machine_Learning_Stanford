function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

x0=0;
for i=1:m
	v=theta'*X(i,:)';
	v=v-y(i);
	x0=x0+v;
end
x0=x0*alpha/m;
x0=theta(1)-x0;

x1=0;
for i=1:m
	v=theta'*X(i,:)'-y(i);
	x1=x1+v*X(i,2);
end
x1=x1*alpha/m;
x1=theta(2)-x1;

theta(1)=x0;
theta(2)=x1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
