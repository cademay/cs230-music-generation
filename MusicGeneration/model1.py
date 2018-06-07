import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Input, LSTM, BatchNormalization, Conv1D
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from encoding_5 import generate_dataset_scheme_4, sequence_length, create_midi_from_results
from keras.callbacks import ModelCheckpoint
from music21 import converter , instrument, note, chord

n_a = 256
#reshapor = Reshape((1, n_values))


def initialize_rnn(X, n_a, n_values):
        model = Sequential()
        
        model.add(Conv1D(128, 3, strides=1, input_shape=(X.shape[1], X.shape[2]), activation = 'relu'))

        #model.add(LSTM(n_a, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
        model.add(Dropout(0.1))
        model.add(LSTM(n_a, return_sequences = True))
        model.add(Dropout(0.3))
        model.add(LSTM(n_a, return_sequences = True))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(n_values))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr = 0.01)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
        return model


def train_rnn(trainFlag):
        X, Y, pitchnames = generate_dataset_scheme_4('dev_small', 3)
        print(X.shape, Y.shape)
        n_values = Y.shape[1]
        model = initialize_rnn(X, n_a, n_values)
        #model.load_weights('weights-improvement-191-0.0000-bigger.hdf5')
        #filepath = "weights_aws-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        if trainFlag:
                #checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 0, save_best_only = True, mode = 'min')

                filepath="models/model1/model1_{epoch:02d}.hdf5"
                callBack = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
                callbackList = [callBack]
                history = model.fit(X, Y, validation_split =.05, epochs = 110, batch_size = 100, callbacks=callbackList)



        return model, X, pitchnames, n_values, history

def generate_music(model, X, pitchnames, n_values):
        #model.load_weights('weights_aws-improvement-123-0.0117-bigger.hdf5')
        start_ind = np.random.randint(0, len(X) -1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        gen_song = X[start_ind]
        pred_out = []
        for n_ind in range(300):
                pred_in = np.reshape(gen_song, (1, len(gen_song), 1))
                pred_in = pred_in/float(n_values)
                pred = model.predict(pred_in, verbose= 0)
                result = int_to_note[np.argmax(pred)]
                pred_out.append(result)

                gen_song = np.append(gen_song, n_ind)
                gen_song = gen_song[1:len(gen_song)]

        return pred_out



def main():
        model, X, pitchnames, n_values, history = train_rnn(True)
        """
        for ii in range(10):
                results = generate_music(model, X, pitchnames, n_values)
                print ("creating midi file no. ", ii)
                create_midi_from_results(results, 'music_outputs/model1/fuego_flames{0}.mid'.format(ii))
                print ('midi created')
        """

main()

