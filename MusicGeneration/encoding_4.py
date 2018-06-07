# Our fourth go at an encoding scheme 
from music21 import converter , instrument, note, chord, stream
import numpy as np
from keras.utils import np_utils
import os


DATA_DIR = 'dev'
DEV2_DIR = 'Dev2'
TEST_DIR = 'test'
CACHE_DIR = 'cache'
notes = [] 
sequence_length = 100


def generate_dataset_scheme_4(dir, m =8):
	# based on an encoding/decoding scheme using music21 from Sigurour Skuli's blog Towards Data Science
	if dir == 'dev':
		directory = DATA_DIR
	if dir == 'Dev2': 
		directory = DEV2_DIR
	else: 
		directory = TEST_DIR
	notes = []
	count = 0 ;
	for filename in os.listdir(directory): 
		if filename.endswith('.mid'): 
			print("generating midi from ", filename)
			midi = converter.parse('{}/{}'.format(directory, filename))
			note_vec = None
			parts = instrument.partitionByInstrument(midi)

			if parts: #file has instrument parts 
				note_vec = parts.parts[0].recurse()
			else: 
				note_vec = midi.flat.notes
			for mus_obj in note_vec: 
				if isinstance(mus_obj, note.Note): 
					notes.append(str(mus_obj.pitch))
				elif isinstance(mus_obj, chord.Chord):
					notes.append('.'.join(str(n) for n in mus_obj.normalOrder))
			count = count + 1

			if count == m: 
				break

	pitchnames = sorted(set(item for item in notes))

			# match pitches to ints 
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	X = []
	Y = []

	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i+sequence_length]
		X.append([note_to_int[char] for char in sequence_in])
		Y.append(note_to_int[sequence_out])

	#print (len(X))

	n_patterns = len(X)

	X = np.reshape(X, (n_patterns, sequence_length, 1))
	# normalize input 
	
	Y = np_utils.to_categorical(Y)
	#X = X/float(Y.shape[1]) - (Y.shape[1])/2
	X = X/float(Y.shape[1])
	return X, Y, pitchnames

def create_midi_from_results(prediction_output, fp = 'test.mid'):
	offset = 0
	prev_note = None
	prev_chord = None 
	#for postprocessing if we need
	
	notes_out = [] 
	for pattern in prediction_output:
	    if ('.' in pattern) or pattern.isdigit():
	        notes_in_chord = pattern.split('.')
	        notes = []
	        for current_note in notes_in_chord:
	            curr_note = note.Note(int(current_note))
	            curr_note.storedInstrument = instrument.Piano()
	            notes.append(curr_note)
	        curr_chord = chord.Chord(notes)
	        curr_chord.offset = offset
	        notes_out.append(curr_chord)
	     
	    # pattern is a note
	    else:
	        curr_note = note.Note(pattern)
	        curr_note.offset = offset
	        curr_note.storedInstrument = instrument.Piano()
	        if curr_note != prev_note: 
		        notes_out.append(curr_note)
		        prev_note = curr_note
	       
	    # increase the offset each iteration....
	    offset += 0.5
	    midi_stream = stream.Stream(notes_out)
	    midi_stream.write('midi', fp)



def main():
	generate_dataset_scheme_4('dev', 3)

if __name__== "__main__":
  main()