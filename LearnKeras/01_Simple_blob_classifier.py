# Imports
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os

# Setting the logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Read the downloaded dataset into X,y
X, y = make_blobs(n_samples=10000, centers=2, random_state=42)
print("X-Shape: ", X.shape, "y-shape: ", y.shape)

# Split using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("X-Shape: ", X_train.shape, "y-shape: ", y_train.shape, "X-Shape: ", X_test.shape, "y-shape: ", y_test.shape)

# Initialize Keras Sequential model
model = Sequential()

# Adding only a Dense layer with Sigmoid activation
model.add(Dense(1, input_shape=(2,), activation = "sigmoid"))
# Using Adam Optimizer with Binary Crossentropy loss function
model.compile(Adam(lr=0.002), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, callbacks=[EarlyStopping(patience=5)], verbose=1)

# Evaluation of test set
evaluated_result = model.evaluate(X_test, y_test)

print("\n\nEvaluated loss: ", evaluated_result[0], "\nEvaluated accuracy: ", evaluated_result[1])