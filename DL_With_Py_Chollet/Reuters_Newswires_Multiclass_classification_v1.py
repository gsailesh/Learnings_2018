
# Import & Data definitions
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE=512
EPOCHS=20
NUM_CLASSES=46
TYPE={'CATEGORICAL':'categorical_crossentropy','SPARSE':'sparse_categorical_crossentropy'}
# Load data
(X_train,y_train),(X_test,y_test) = reuters.load_data(num_words=10000)


# Decode data
word_index = None; reverse_word_index = None

def decode_sample(datapoint, word_index=None, reverse_word_index=None):
	
	if word_index is None:
		word_index = reuters.get_word_index()
	
	if reverse_word_index is None:
		reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])

	text = ' '.join(reverse_word_index.get(i-3,'?') for i in datapoint)
	return text

print(decode_sample(X_train[1], word_index, reverse_word_index))



# Preprocess
def preprocess(X,y,dims=10000, type=TYPE['CATEGORICAL']):
	
	X_result = np.zeros((len(X),dims))
	
	for i,v in enumerate(X):
		X_result[i,v] = 1

	if type.lower() == TYPE['SPARSE'].lower():
		y_result = np.array(y)
	else:
		y_result = to_categorical(y)

	return (X_result, y_result)


X_train_vec, y_train_vec = preprocess(X_train,y_train, type=TYPE['SPARSE'])
X_test_vec, y_test_vec = preprocess(X_test,y_test, type=TYPE['SPARSE'])

# Model init
model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(10000,)))
model.add(Dense(128,activation='relu'))
model.add(Dense(46,activation='softmax'))

# Model compile
model.compile(optimizer=Adam(lr=0.002),loss=TYPE['SPARSE'],metrics=['accuracy'])

# Model fit
history = model.fit(X_train_vec, y_train_vec, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test_vec,y_test_vec))

# Training visualization
epoch_range = range(1,EPOCHS+1,1)

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

training_accuracy = history.history['acc']
validation_accuracy = history.history['val_acc']

plt.plot(epoch_range,training_loss,'b',label='Training loss')
plt.plot(epoch_range, validation_loss, 'r', label='Validation loss')
plt.title('Training v Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() # clear the figure

plt.plot(epoch_range,training_accuracy,'b',label='Training accuracy')
plt.plot(epoch_range, validation_accuracy, 'r', label='Validation accuracy')
plt.title('Training v Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

