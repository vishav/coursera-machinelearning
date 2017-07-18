function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
hypothesis  = X*theta;
diff = hypothesis - y;
J = (1/(2*m))*diff'*diff;
%J = (1/(2*m))*sum((hypothesis -y).^2);

end
