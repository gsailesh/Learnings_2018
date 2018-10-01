# Imports
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Data read and train/test split
X,y = make_circles(n_samples=10000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Layer definitions - Functional API way
inputs = Input(shape=(2,))
hidden_layers = Dense(4, activation="tanh", name="HiddenLayer-1")(inputs)
hidden_layers = Dense(4, activation="tanh", name="HiddenLayer-2")(hidden_layers)
output = Dense(1, activation="sigmoid", name="OutputLayer")(hidden_layers)

model = Model(inputs=inputs, outputs=output)
# Display Model summary
print(model.summary())

model.compile(optimizer=Adam(lr=0.002), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=500, callbacks=[EarlyStopping(monitor='accuracy', patience=5, mode=max)], verbose=1)

evaluated_result = model.evaluate(X_test, y_test)
print("\n\nEvaluation loss: ", evaluated_result[0], "\nEvaluation accuracy: ", evaluated_result[1])
