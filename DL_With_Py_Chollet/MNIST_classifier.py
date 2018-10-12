from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

im_dims = (28,28)
(X_train, y_train),(X_test, y_test) = mnist.load_data()
# print(X_train.shape)

def preprocess_mnist(X, y, dims):
    
    X_len = X.shape[0]
    y_len = y.shape[0]

    X = X.reshape((X_len, dims[0] * dims[1]))
    
    X = X.astype('float32') / 255
    y = to_categorical(y)

    return (X,y)

X_train_norm, y_train_norm = preprocess_mnist(X_train, y_train, im_dims)
X_test_norm, y_test_norm = preprocess_mnist(X_test, y_test, im_dims)

model = Sequential()
model.add(Dense(units=512,activation='relu',input_shape=(im_dims[0]*im_dims[1],)))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer = Adam(lr=0.002),loss='categorical_crossentropy',metrics=['accuracy'])
print("X_train_norm: ", X_train_norm.shape)
model.fit(X_train_norm,y_train_norm,batch_size=64,epochs=10)

evaluation_result = model.evaluate(X_test_norm, y_test_norm)

print("Evaluation result (Loss & Accuracy): ", evaluation_result[0], evaluation_result[1])