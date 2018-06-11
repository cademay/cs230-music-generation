from keras.models import load_model
import numpy as np
from music21 import converter , instrument, note, chord, stream
import numpy as np
from keras.utils import np_utils
import os

sequence_length = 100


data_file = 'dev_small'
number_of_songs = 1

def create_midi_from_results(prediction_output, fp = 'test.mid'):
    offset = 0
    
    prev_note = None
    prev_chord = None

    notes_out = []
    
    for pattern in prediction_output:
        # pattern is a chord
        pattern, dur = pattern.split('_')
         
        if dur == 'complex' or dur == 'zero' or dur == 'breve':
            continue
             
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                curr_note = note.Note(int(current_note))
                curr_note.storedInstrument = instrument.Piano()
                notes.append(curr_note)
            curr_chord = chord.Chord(notes)
            curr_chord.offset = offset
            curr_chord.duration.type = dur 
            notes_out.append(curr_chord)
         
        # pattern is a note
        else:
            
            
            curr_note = note.Note(pattern)
            curr_note.offset = offset
            curr_note.storedInstrument = instrument.Piano()
            curr_note.duration.type = dur
            if curr_note != prev_note: 
                notes_out.append(curr_note)
                prev_note = curr_note
           
        # increase offset to avoid stacking notes. 
        dur_to_off = {'whole':4.0, 'half':2.0, 'quarter':1.0, 'eighth':0.5, '16th':0.25}
       
        offset += dur_to_off[dur]
        midi_stream = stream.Stream(notes_out)
        midi_stream.write('midi', fp)


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

                gen_song = np.append(gen_song, np.argmax(pred))
                gen_song = gen_song[1:len(gen_song)]

        return pred_out


# get data
X, Y, pitchnames = data_preprocessing()
n_values = Y.shape[1]


# load model
model= load_model("model11_100.hdf5")
print("params: ", model.count_params())


# generate music
for ii in range(1):
	    results = generate_music(model, X, pitchnames, n_values)
	    print ("creating midi file no. ", ii)
	    create_midi_from_results(results, 'music_outputs/model1/model11_500__{0}.mid'.format(ii))
	    print ('midi created')


# calculate loss
model_load.evaluate(X, Y, batch_size = 100)


