# Imports
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keras.backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 30
IMAGE_SIZE = (28,28)

(X_train, y_train),(X_test, y_test) = fashion_mnist.load_data()

def format_data(data, format, dims):
    
    if format == 'channels_first':
        data = data.reshape(data.shape[0], 1, dims[0], dims[1])
        input_shape = (1, dims[0], dims[1])
    else:
        data = data.reshape(data.shape[0], dims[0], dims[1], 1)
        input_shape = (dims[0], dims[1], 1)
    
    return (data, input_shape)


def normalize_data(data):
    
    data = data.astype('float32')
    data /= 255

    return data


(X_train, input_shape) = format_data(X_train, K.image_data_format(), IMAGE_SIZE)
(X_test, _) = format_data(X_test, K.image_data_format(), IMAGE_SIZE)

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)

#One-hot encoding
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax"))

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=['accuracy'])
model_history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor='accuracy',patience=4,mode=max)])

evaluated_result = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print("\n\nTest accuracy: ", evaluated_result[1], "\nTest loss: ", evaluated_result[0])

epoch_list = list(range(1,len(model_history.history['acc'])+1))
plt.plot(epoch_list, model_history.history['acc'], epoch_list, model_history['val_accs'])
plt.legend(['Training_accuracy', 'Validation_accuracy'])
plt.show()