
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error

diabetes_data = datasets.load_diabetes()

diabetes_X = diabetes_data.data
diabetes_y = diabetes_data.target

X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]

y_train = diabetes_y[:-20]
y_test = diabetes_y[-20:]

# Initialize parameters
#----------------------
# W: should be of same dimension as number of features in X ~ 10
# Shape of diabetes dataset is obtained by .shape() -- [rows, cols]; hence choosing index 1
#
# b: scalar for bias -- randomly initialized
#
# learning_rate: randomly initialized to a small value
#
# epoch: represents the number of iterations
#---------------------------------------------
N = X_train.shape[0]
W = np.random.uniform(low=-0.1,high=0.1,size=diabetes_X.shape[1])
b = 0.1
learning_rate = 0.00005
epoch = 10000

# Estimate y_cap to calculate the cost function and apply Gradient Descent Optimization
# y_cap = np.zeros(y_train)

for iter in range(epoch):
	y_cap = X_train.dot(W) + b
	error = y_cap - y_train
	cost = np.mean(np.power(error,2))

	grad_W = -(1/N)*error.dot(X_train)
	grad_b = -(1/N)*np.sum(error)

	W = W - learning_rate*grad_W
	b = b - learning_rate*grad_b

	if iter % 1000 == 0: print("Epoch %d: MSE: %f" % (iter, cost))


print("Final Weights: ", W)

# Validation/Testing
y_val = X_test.dot(W) + b

val_error = y_val - y_test
val_mse = np.mean(np.power(val_error,2))

print("Validation MSE: %.2f" % val_mse)
	
