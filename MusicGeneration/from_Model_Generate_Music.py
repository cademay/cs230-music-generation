from keras.models import load_model
import numpy as np
from encoding_5 import generate_dataset_scheme_4, sequence_length, create_midi_from_results


X,Y,pitchnames = generate_dataset_scheme_4('dev_small', 1)
n_values = Y.shape[1]

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

model_load = load_model("models/model1_100.hdf5")
model_load_weights = model_load.get_weights()



for ii in range(2):
	    results = generate_music(model_load, X, pitchnames, n_values)
	    print ("creating midi file no. ", ii)
	    create_midi_from_results(results, 'music_outputs/model1/model1_output{0}.mid'.format(ii))
	    print ('midi created')


