from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

im_dims = (28,28)

BATCH_SIZE = 32
EPOCHS = 5

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
history = model.fit(X_train_norm,y_train_norm,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(X_test_norm, y_test_norm))

evaluation_result = model.evaluate(X_test_norm, y_test_norm)

print("Evaluation result (Loss & Accuracy): ", evaluation_result[0], evaluation_result[1])



# Show sample
index = 12
plt.imshow(X_test[index],cmap=plt.cm.binary)
plt.show()

# predicted_sample = model.predict(np.asarray([X_test_norm[1]]),verbose=1)
# print("Sample class: ",predicted_sample[0].argmax(axis=0),"\tSample score: ",max(predicted_sample[0]))

predictions = model.predict(np.asarray([X_test_norm[index]]),verbose=1)
sample_class = predictions[0].argmax(axis=0); sample_score = predictions[0][sample_class]
print("Sample class: ",sample_class,"\tSample score: ",sample_score)

epoch_range = range(1,EPOCHS+1,1)

# Training & Evaluation loss
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

plt.plot(epoch_range,training_loss,'r',label='Training loss')
plt.plot(epoch_range,validation_loss,'b',label='Validation loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# Training & Evaluation Accuracy
training_acc = history.history['acc']
validation_acc = history.history['val_acc']

plt.plot(epoch_range,training_acc,'r',label='Training accuracy')
plt.plot(epoch_range,validation_acc,'b',label='Validation accuracy')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.show()
