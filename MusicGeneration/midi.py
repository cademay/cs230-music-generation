import mido
import numpy as np


NUM_NOTES = 128
NUM_TIME_SHIFTS = 100
TICK_PER_SECOND = 1000
TICKS_PER_BEAT = 5000
DEFAULT_VELOCITY = 50


def decode_midi_1(midi, resolution):
    decoding = []
    bucket = np.zeros(2*NUM_NOTES)
    time = 0
    for message in midi:
        time += message.time
        if time > resolution:
            numSkips = int(time/resolution)
            time -= resolution * numSkips
            decoding.append(bucket)
            bucket = np.zeros(2*NUM_NOTES)
            for _ in range(numSkips-1):
                decoding.append(np.zeros(2*NUM_NOTES))
        if message.type == 'note_on':
            if message.velocity == 0:
                bucket[NUM_NOTES + message.note] = 1
            else:
                bucket[message.note] = 1
        # if message.type not in ['note_on', 'control_change', 'program_change', 'sequencer_specific', 'set_tempo', 'time_signature', 'sysex', 'polytouch', 'end_of_track']:
        #     print 'Warning: Unknown message type ({}).'.format(message.type)
    decoding.append(bucket)
    return np.array(decoding)


def encode_midi_1(decoding, track, resolution):
    offset = 0
    for row in decoding:
        offset += int(TICK_PER_SECOND*resolution)
        for i in np.where(row == 1)[0]:
            if i < NUM_NOTES:
                track.append(mido.Message('note_on', note=i, velocity=DEFAULT_VELOCITY, time=offset))
            else:
                track.append(mido.Message('note_on', note=i-NUM_NOTES, velocity=0, time=offset))
            offset = 0


def decode_midi_2(midi, resolution):
    decoding = []
    emptyBucket = np.zeros(2*NUM_NOTES + NUM_TIME_SHIFTS)
    bucket = emptyBucket.copy()
    time = 0
    counter = 0
    totaltime = 0
    for message in midi:
        time += message.time
        totaltime += message.time
        if totaltime > 5: return np.array(decoding)
        if time > resolution:
            if np.sum(bucket) > 0:
                bucket[2*NUM_NOTES + counter] = 1
                # print np.where(bucket[:NUM_NOTES])[0], np.where(bucket[NUM_NOTES:2*NUM_NOTES])[0], counter*.01
                decoding.append(bucket)
                bucket = emptyBucket.copy()
                counter = -1
            numSkips = int(time/resolution)
            counter += numSkips
            time -= resolution * numSkips
        if message.type == 'note_on':
            if message.velocity == 0:
                bucket[NUM_NOTES + message.note] = 1
            else:
                bucket[message.note] = 1
        if counter >= NUM_TIME_SHIFTS:
            numFullShifts = int(counter/float(NUM_TIME_SHIFTS))
            for _ in range(numFullShifts):
                temp = emptyBucket.copy()
                temp[-1] = 1
                # print tuple([]), NUM_TIME_SHIFTS*.01
                decoding.append(temp)
            counter -= numFullShifts * NUM_TIME_SHIFTS
    bucket[2*NUM_NOTES + int(time/resolution)] = 1
    # print np.where(bucket[:NUM_NOTES])[0], (int(time/resolution) + 1)*.01
    decoding.append(bucket)
    return np.array(decoding)


def encode_midi_2(decoding, track, resolution):
    offset = 0
    for row in decoding:
        timeShift = np.where(row[2*NUM_NOTES:] == 1)[0][0]+1
        offset += int(TICK_PER_SECOND*resolution) * timeShift
        print (offset)
        for i in np.where(row[:2*NUM_NOTES] == 1)[0]:
            if i < NUM_NOTES:
                track.append(mido.Message('note_on', note=i, velocity=DEFAULT_VELOCITY, time=offset))
            else:
                track.append(mido.Message('note_on', note=i-NUM_NOTES, velocity=0, time=offset))
            offset = 0

# 
# def decode_midi_3(midi, resolution):
#     decoding = decode_midi_2(midi, resolution)
#     newDecoding = []
#     notesOn = []
#     prevNotesOn = []
#     for row in decoding:
#         bucket = np.zeros(NUM_NOTES + NUM_TIME_SHIFTS)
#         bucket[NUM_NOTES:] = row[2*NUM_NOTES:]
#         newNotesOn = np.where(row[:NUM_NOTES] == 1)[0]
#         newNotesOff = np.where(row[NUM_NOTES:2*NUM_NOTES] == 1)[0]
#         prevNotesOn = list(notesOn)
#         notesOn = list(set(prevNotesOn) | set(newNotesOn) - set(newNotesOff))
#         bucket[notesOn] = 1
#         newDecoding.append(bucket)
#     return np.array(newDecoding)
#
# def encode_midi_3(decoding, track, resolution):
#     newDecoding = []
#     notesOn = []
#     prevNotesOn = []
#     for row in decoding:
#         bucket = np.zeros(2*NUM_NOTES + NUM_TIME_SHIFTS)
#         prevNotesOn = list(notesOn)
#         notesOn = np.where(row[:NUM_NOTES] == 1)[0]
#         newNotesOff = list(set(prevNotesOn) - set(notesOn))
#         newNotesOn = list(set(notesOn) - set(prevNotesOn))
#         if len(newNotesOn) > 0:
#             bucket[newNotesOn] = 1
#         if len(newNotesOff) > 0:
#             bucket[NUM_NOTES + np.array(newNotesOff)] = 1
#         bucket[2*NUM_NOTES:] = row[NUM_NOTES:]
#         newDecoding.append(bucket)
#     encode_midi_2(np.array(newDecoding), track, resolution)

# def decode_midi_3(midi, resolution):
#     decoding = []
#     emptyBucket = np.zeros(NUM_NOTES + NUM_TIME_SHIFTS)
#     bucket = emptyBucket.copy()
#     time = 0
#     counter = 0
#     totaltime = 0
#     prevBucket = emptyBucket.copy()
#     for message in midi:
#         time += message.time
#         totaltime += message.time
#         if totaltime > 5: return np.array(decoding)
#         if time > resolution:
#             if np.any(prevBucket != bucket):
#                 bucket[NUM_NOTES + counter] = 1
#                 # print np.where(bucket[:NUM_NOTES])[0], counter*.01
#                 decoding.append(bucket.copy())
#                 prevBucket = bucket.copy()
#                 bucket[NUM_NOTES + counter] = 1
#                 counter = -1
#             numSkips = int(time/resolution)
#             counter += numSkips
#             time -= resolution * numSkips
#         if message.type == 'note_on':
#             if message.velocity > 0:
#                 bucket[message.note] = 1
#             else:
#                 bucket[message.note] = 0
#         if counter > NUM_TIME_SHIFTS:
#             numFullShifts = int(counter/float(NUM_TIME_SHIFTS))
#             for _ in range(numFullShifts):
#                 bucket[-1] = 1
#                 # print tuple([]), NUM_TIME_SHIFTS*.01
#                 decoding.append(bucket.copy())
#                 bucket[-1] = 0
#             counter -= numFullShifts * NUM_TIME_SHIFTS
#     bucket[NUM_NOTES + int(time/resolution)] = 1
#     # print np.where(bucket[:NUM_NOTES])[0], (int(time/resolution) + 2)*.01
#     decoding.append(bucket)
#     return np.array(decoding)
#
#
# def encode_midi_3(decoding, track, resolution):
#     offset = 0
#     notesOn = []
#     print 'asdkfjlaksjdfklajskldfjalsjdflkasjkl'
#     for row in decoding:
#         # print np.where(row[:NUM_NOTES] == 1)[0], np.where(row[NUM_NOTES:] == 1)[0][0]*.01
#         timeShift = np.where(row[NUM_NOTES:] == 1)[0][0]+1
#         offset += int(TICK_PER_SECOND*resolution) * timeShift
#         print offset
#         prevNotesOn = list(notesOn)
#         notesOn = []
#         for i in np.where(row[:NUM_NOTES] == 1)[0]:
#             if not i in prevNotesOn:
#                 track.append(mido.Message('note_on', note=i, velocity=DEFAULT_VELOCITY, time=offset))
#                 offset = 0
#                 print i
#             notesOn.append(i)
#         for note in list(set(prevNotesOn) - set(notesOn)):
#             track.append(mido.Message('note_on', note=note, velocity=0, time=offset))
#             offset = 0
#             print i


def decode_midi(filename, version=2, resolution=.01):
    midi = mido.MidiFile(filename)
    if version is 1:
        return decode_midi_1(midi, resolution)
    elif version is 2:
        return decode_midi_2(midi, resolution)
    elif version is 3:
        return decode_midi_3(midi, resolution)
    return None


def encode_midi(decoding, filename, version=2, resolution=.01):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    if version is 1:
        encode_midi_1(decoding, track, resolution)
    elif version is 2:
        encode_midi_2(decoding, track, resolution)
    elif version is 3:
        encode_midi_3(decoding, track, resolution)
    midi.save(filename)


def main():
    # encode_midi(decode_midi('testFiles/ADIG03.mid',1, resolution=.01),'test1.mid',1, resolution=.01)
    a = mido.MidiFile('testFiles/ADIG03.mid')
    time = 0
    for m in a:
        time += m.time
        if time < 4 and m.type == 'note_on' and m.velocity > 0:
            print (m.note, round(m.time, 2))
    print ('-'*80)
    print ('Encoding 2:')
    encode_midi(decode_midi('testFiles/ADIG03.mid',2, resolution=.01),'test2.mid',2, resolution=.01)
    print ('-'*80)
    print ('Encoding 3')
    encode_midi(decode_midi('testFiles/ADIG03.mid',3, resolution=.01),'test3.mid',3, resolution=.01)


if __name__== "__main__":
  main()
