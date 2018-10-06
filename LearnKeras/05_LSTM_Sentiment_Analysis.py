import os
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

NUM_WORDS = 6000 # Most frequent words to be considered
SKIP_TOP = 2 # Skipping the most frequent words like a,an,the
MAX_REVIEW_LENGTH = 150

BATCH_SIZE = 24
EPOCH = 2

# Loading pre-processed sentiment analysis data
(X_train, y_train), (X_test, y_test)= imdb.load_data(num_words=NUM_WORDS, skip_top=SKIP_TOP)

# Pad/truncate word sequences to maintain consistency in their length
X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)

print("X_train shape: ", X_train.shape, "X_test shape: ", X_test.shape)

model = Sequential()
model.add(Embedding(NUM_WORDS, 64))
model.add(LSTM(64, recurrent_dropout=0.3, dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.002), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, callbacks=[EarlyStopping(monitor='val_acc', patience=2, mode='max')], validation_data=(X_test, y_test))

score, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("Score: ", score, "Accuracy: ", accuracy)