from midi import encode_midi, decode_midi
import numpy as np
import os


DATA_DIR = 'dev'
TEST_DIR = 'test'
CACHE_DIR = 'cache'


def generate_dataset(dir, timesteps):
    if dir == 'dev':
        directory = DATA_DIR
    else:
        directory = TEST_DIR
    try:
        np.load(directory + '_CACHE.npz')
    except:
        pass
    dataset = []
    for filename in os.listdir(directory):
        decoding = decode_midi('{}/{}'.format(DATA_DIR, filename))
        for subset in range(decoding.shape[0] / timesteps):
            subDecoding = decoding[subset * timesteps:]
            if subDecoding.shape[0] < timesteps:
                continue
            dec = subDecoding[:timesteps]
            dataset.append(dec.copy().T)
    return dataset


def generate_dataset_iterator(dir):
    if dir == 'dev':
        directory = DATA_DIR
    else:
        directory = TEST_DIR
    try:
        npz = np.load(directory + '_CACHE.npz')
        arr = npz[npz.files[0]]
        for decoding in arr:
            yield decoding
    except GeneratorExit:
        pass
    except:
        for filename in os.listdir(directory):
            if filename.endswith('.mid'):
                yield decode_midi('{}/{}'.format(directory, filename))


def save_decoding(decoding, filename):
    encode_midi(decoding, filename)


def main():
    counter = 0
    allx = []
    for x in generate_dataset_iterator('dev'):
        counter += 1
        print( counter)
        allx.append(x)
    np.savez(DATA_DIR + '_CACHE.npz', allx)
    allx = []
    for x in generate_dataset_iterator('test'):
        allx.append(x)
    np.savez(TEST_DIR + '_CACHE.npz', allx)

if __name__== "__main__":
  main()
