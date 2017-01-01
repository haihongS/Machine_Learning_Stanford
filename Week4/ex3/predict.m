function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X: 5000*400

X=[ones(m,1) X];
n=size(Theta1,1);
a1=zeros(m,n);
size(a1)
for i=1:m
	for j=1:n
		a1(i,j)=sigmoid(Theta1(j,:)*X(i,:)');
	end
end


m=size(a1,1);
a1=[ones(m,1) a1];
n=size(Theta2,1);
v=zeros(m,n);

for i=1:m
	for j=1:n
		v(i,j)=sigmoid(Theta2(j,:)*a1(i,:)');
	end
end

[q,p]=max(v,[],2);







% =========================================================================


end
