import numpy as np

'''
A naive implementation of RNN
'''

# Initialize timesteps & sequence length
timesteps = 100
num_features = 32
num_output_features = 64

# Initialize input & state matrix
input_matrix = np.random.random((timesteps, num_features))
state_at_timestep = np.zeros((num_output_features,))

# Initialize Weight matrices and Bias
W = np.random.random((num_output_features, num_features))
U = np.random.random((num_output_features,num_output_features))
bias = np.random.random((num_output_features,))

outputs_list = []

for input_t in input_matrix:
	# for every timestep, take the input and apply the activation

	'''
		shape = (num_output_features,num_features)*(num_features,) +
				(num_output_features,num_output_features)*(num_output_features,) +
				(num_output_features,)

			  = (num_output_features,) +
			  	(num_output_features,) +
			  	(num_output_features,)

			  = (num_output_features,)
	'''
	activation = np.tanh(np.dot(W,input_t) + np.dot(U,state_at_timestep) + bias)

	# append the activation output to the `outputs_list`
	outputs_list.append(activation)

	# `state_at_timestep` would be updated to the current activation output
	state_at_timestep = activation

final_output_sequence = np.concatenate(outputs_list,axis=0) # shape = (timesteps,num_output_features)
print('Final output sequence shape: {0}'.format(final_output_sequence.shape))