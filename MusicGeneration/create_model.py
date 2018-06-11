import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from music21 import converter , instrument, note, chord
from music21 import converter , instrument, note, chord, stream
import numpy as np
from keras.utils import np_utils
import os


# which model do you want to train?
from model3 import create_model

n_a = 256
sequence_length = 100

data_file = 'Datasets/Dataset_44_Eflatmaj/Dev'
number_of_songs = 1


def data_preprocessing():
    
    # “How to Generate Music Using a LSTM Neural Network in Keras.”
    # -- Towards Data Science, Medium, 7 Dec. 2017. Skúli, Sigurður. 
    
    # “LSTiestoM: Generating Classical Music.”
    # D. Gallegos and S. Metzger. CS230: Deep Learning, Winter 2018. 

    directory = data_file
    notes = []
    count = 0 ;
    
    m = number_of_songs

    # for each song
    for filename in os.listdir(directory): 
        if filename.endswith('.mid'): 
            
            print("parsing midi input from ", filename)
            midi = converter.parse('{}/{}'.format(directory, filename))
            
            notes_vec = None
            notes_vec = midi.flat.notes

            for mus_obj in notes_vec: 
                if isinstance(mus_obj, note.Note): 
                    str1 = str(mus_obj.pitch)
                    str3 = '_'
                    str2 = str((mus_obj.duration.type))
                    n_s = str1 + str3 +  str2
                    notes.append(n_s)
                elif isinstance(mus_obj, chord.Chord):
                    str1 = ('.'.join(str(n) for n in mus_obj.normalOrder))
                    str2 = str((mus_obj.duration.type))
                    notes.append(str1 + '_' +  str2)
                    
            count += 1

            if count == m: 
                break

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    X = []
    Y = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i+sequence_length]
        X.append([note_to_int[char] for char in sequence_in])
        Y.append(note_to_int[sequence_out])


    n_patterns = len(X)
    X = np.reshape(X, (n_patterns, sequence_length, 1))
    
    # normalization
    Y = np_utils.to_categorical(Y)
    X = X/float(Y.shape[1])
    
    return X, Y, pitchnames



def build_model():
    
        # get data
        X, Y, pitchnames = data_preprocessing()
        n_values = Y.shape[1]

        # create a model
        model = create_model(X, n_a, n_values)
        
        # make sure model saves
        filepath="models/model1/model1_{epoch:02d}.hdf5"
        callBack = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
        callbackList = [callBack]
        
        # train model
        model.fit(X, Y, validation_split =.05, epochs = 110, batch_size = 100, callbacks=callbackList)


def main():
    build_model()
        

main()

