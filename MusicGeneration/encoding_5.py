# Our fourth go at an encoding scheme 
from music21 import converter , instrument, note, chord, stream
import numpy as np
from keras.utils import np_utils
import os


DATA_DIR = 'dev'
DEV2_DIR = 'Dev2'
TRAIN_DIR = 'train'

TEST_DIR = 'train'


CACHE_DIR = 'cache'
notes = [] 
sequence_length = 100


def generate_dataset_scheme_4(dir, m =8):
	# based on an encoding scheme using music21 from Sigurour Skuli's blog Towards Data Science
	if dir == 'dev':
		directory = DATA_DIR
	if dir == 'Dev2': 
		directory = DEV2_DIR
	#if dir == 'train':
     #   directory = TRAIN_DIR
	else: 
		directory = TEST_DIR
	notes = []
	count = 0 ;
	for filename in os.listdir(directory): 
		if filename.endswith('.mid'): 
			print("generating midi from ", filename)
			midi = converter.parse('{}/{}'.format(directory, filename))
			notes_vec = None
			#parts = instrument.partitionByInstrument(midi)

			#if parts: #file has instrument parts 
			#	notes_vec = parts.parts[0].recurse()
			#else: 
			notes_vec = midi.flat.notes
			#print(notes_vec)
			for mus_obj in notes_vec: 
				if isinstance(mus_obj, note.Note): 
					#print (mus_obj.duration, mus_obj.offset)
					str1 = str(mus_obj.pitch)
					str3 = '_'
					str2 = str((mus_obj.duration.type))
					n_s = str1 + str3 +  str2
					notes.append(n_s)
				elif isinstance(mus_obj, chord.Chord):
					str1 = ('.'.join(str(n) for n in mus_obj.normalOrder))
					str2 = str((mus_obj.duration.type))

					notes.append(str1 + str3 +  str2)
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
	#print (notes)
	return X, Y, pitchnames

def create_midi_from_results(prediction_output, fp = 'test.mid'):
	offset = 0
	
	prev_note = None
	prev_chord = None

	notes_out = []
	
	for pattern in prediction_output:
	    # pattern is a chord
	    pattern, dur = pattern.split('_')
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

def main():
	generate_dataset_scheme_4('dev', 1)

if __name__== "__main__":
  main()
