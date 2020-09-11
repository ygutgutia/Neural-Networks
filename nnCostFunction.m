function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

yN=zeros(size(y,1),size(Theta2,1));
X=[ones(m,1) X];

a1=X;
X=X';

z2=Theta1*X;
a2=sigmoid(z2);
a2=a2';
a2=[ones(size(a2,1),1) a2];
a2=a2';
z3=Theta2*a2;
h=sigmoid(z3);
a3=h;
for i=1:m
  l=y(i,1);
  yN(i,l)=1;
endfor
h1=yN.*log(h)'+(1-yN).*log(1-h)';

J1=sum(h1(:));
J1=J1/m;

T1=Theta1.*Theta1;
T2=Theta2.*Theta2;

J2=sum(T1(:))+sum(T2(:));
r1=T1(:,1);
r2=T2(:,1);
J3=sum(r1(:))+sum(r2(:));
J2=J2-J3;
J2=J2*lambda/(2*m);

J=J2-J1;


##disp(size(y)); 5000 1
##disp(size(a2));%26   5000
##   
##disp(size(X)); %401   5000
##    
##disp(size(Theta1));%25   401
##   
##disp(size(Theta2));%10   26

del3=a3'.-y;   %5000*10
del2=(Theta2'*del3').*(a2.*(1-a2));%26*5000

rh1=lambda*Theta1;
rh2=lambda*Theta2;
for i=1:size(Theta1,1)
  rh1(i,1)=0;
endfor

for i=1:size(Theta2,1)
  rh2(i,1)=0;
endfor

Theta1_gradP=(del2*X')/m;
Theta1_grad=Theta1_gradP(2:size(Theta1_gradP,1),size(Theta1_gradP,2));


Theta1_grad=Theta1_grad.+rh1;%25*401
Theta2_grad=(a2*del3)'/m.+rh2;%10*26
grad=[Theta1_grad(:) ; Theta2_grad(:)];












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
