import numpy as np

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Embedding

MAX_FEATURES = 10000
MAX_LEN = 500
BATCH_SIZE = 32
EPOCHS = 1

# Load data
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test)

# rnn_model = Sequential()
# rnn_model.add(Embedding(MAX_FEATURES, 100))
# rnn_model.add(SimpleRNN(100))
# rnn_model.add(Dense(1,activation='sigmoid'))

# rnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# rnn_history = rnn_model.fit(X_train,y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=1)

# print(rnn_history.history)

lstm_model = Sequential()
lstm_model.add(Embedding(MAX_FEATURES, 100))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(1,activation='sigmoid'))

lstm_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
lstm_history = lstm_model.fit(X_train,y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=1)

print(lstm_history.history)