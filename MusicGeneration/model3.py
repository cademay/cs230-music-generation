from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Input, LSTM, BatchNormalization, Conv1D, TimeDistributed
from keras.optimizers import Adam, RMSprop


def initialize_rnn(X, n_a, n_values):
        model = Sequential()

        model.add(LSTM(n_a, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(n_values))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr = 0.01)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
        return model


