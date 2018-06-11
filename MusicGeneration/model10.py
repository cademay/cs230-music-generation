from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Input, LSTM, BatchNormalization, Conv1D, TimeDistributed
from keras.optimizers import Adam, RMSprop




def initialize_rnn(X, n_a, n_values):
        model = Sequential()

        model.add(Conv1D(128, 5, strides=1, input_shape=(X.shape[1], X.shape[2]), activation = 'relu'))
        model.add(Dropout(0.1))


        model.add(LSTM(n_a, return_sequences = True))
        model.add(Dropout(0.1))

        model.add(Conv1D(128, 5, strides=1, activation = 'relu'))
        model.add(Dropout(0.1))

        model.add(TimeDistributed(Dense(n_a)))
        model.add(Dropout(0.1))

        model.add(LSTM(n_a, return_sequences = True))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(n_values))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr = 0.01)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
        return model


