import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from utils import generate_dataset_iterator, save_decoding

# Training Parameters
learning_rate = 0.001
training_steps = 10000

# Network Parameters
timesteps = 1000
num_hidden = 10 # hidden layer num of features
n_values = 128
n_a = 64

# define layers objects. 
reshapor = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation='softmax')

def load_X_train(): 
    X = np.zeros((1,1))
    for decoding in generate_dataset_iterator('dev'): 
        print (decoding.shape)
    return X

def load_X_dev(): 
    X = np.zeros((1,1))
    for decoding in generate_dataset_iterator('test'): 
        print (decoding.shape)
    return X


def load_Y(X): 
    print (X)
    return X


def initialize_nn(Tx, n_a, n_values):
    #inputs: 
    #Tx -- length of sequence
    #n_a is number of activations used in our model 
    #n_values - number of notes we're looking at. 

    #outputs: 
    #model -- a keras model. 

    X = Input(shape= (Tx, n_values))

    #define initial hiddnen state for decoder LSTM. 
    a0 = Input(shape = (n_a,), name = 'a0')
    c0 = Input(shape = (n_a,), name = 'c0')
    a = a0
    c = c0 

    #initialize empty list to append the outputs. 
    outputs = []

    #LoopSZN 
    for t in range(Tx): 

        #select teeth time step vector from X. 
        x = Lambda(lambda x: X[:,t,:])(X)
        #use reshapor to reshape x 
        x = reshapor(x)
        #do a step of the LSTM cell. 
        a, _, c = LSTM_cell(x, initial_state = [a,c])
        #apply densor. 
        out = densor(a)
        # add to the list of outputs
        outputs.append(out)

    #create the model and return it @kateupton
    model = Model([X, a0, c0], outputs)
    return model

def beat_maker_model(LSTM_cell, densor, n_values, n_a, Ty): 

    #inputs: LSTM cell - trained LSTM cell. 
    #densor: trained densor from model 
    # n_values = integer, number of MIDI values. 
    # n_a = number of hidden units in the LSTM cell. 
    # Ty = number of time steps to generate. 

    #outputs: beat_maker_model -- Keras mdoel instance 
    x0 = Input(shape = (1,n_values))

    #define s0, initial hidden state for the decoder LSTM. 
    a0 = Input(shape = (n_a,), name='a0')
    c0 = Input(shape=(n_a,), name = 'c0')
    a = a0
    c = c0 
    x = x0 

    # empty list of outputs
    outputs = []

    #loop over Ty and gen a value at every timestep
    for t in range(Ty): 

        #one step of LSTM_cell 
        a, _, c = LSTM_cell(x, initial_state=[a,c])
        out = densor(a)
        outputs.append(out)
        X = Lambda(one_hot)(out)

    beat_maker = Model([xo, a0, c0], outputs)
    return beat_maker


def predict_and_sample(beat_maker, x_initializer, a_initializer, c_initializer): 

    """
    Inputs: beat_maker: keras model .
    x - one hot vector, iunitialzies. 
    a - initializes hidden states. 
    c - initializes cell state of LSTM cell 

    outputs: 
    results, numpy array of Ty, n_values, one hot vectors which are the values generated. 
    indices - numpy-array of shape (Ty, 1), matrix of indices presenting the values generated. 
    """
    # get an output from our model
    pred = beat_maker.predict([x_initializer, a_initializer, c_initializer])
    #make it into an array of indices. 
    indices = np.argmax(pred, axis = 2)
    # now covert into one hot vectors. 
    results = to_categorical(indices, num_classes = n_values)



def run_nn(): 

    #TODO: load these with Gallegos' code. 
    #X = load_X  
    #Y = load_Y #what exactly is this? 
    X = load_X_train()
    Y = load_Y(X)
    X_dev = load_X_dev()
    Y_dev = load_Y(X_dev)


    #define some more variables
    Tx = 100 # number of timesteps. can be set to a length.
    model = initialize_nn(Tx, n_a, n_values)
    opt  = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay=0.01)
    print ("compile start")
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])
    print ("compile end")
    # set number of training examples. 
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    print ("here")
    model.fit([X, a0, c0], list(Y), epochs=100)
    #loss, acc = model.evaluate(X_dev, Y_dev)
    # print("dev set accuracy = ", acc)


    Ty = 100; 

    #now its time to generate some beats.
    beat_maker = beat_maker_model(LSTM_cell, densor, n_values, n_a, Ty)

    #now, make the inital x and LSTM variables a and c. 
    x_initilaizer = np.zeros((1, 1, n_values))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    #finally, predict and smaple. 
    results, indices = predict_and_sample(beat_maker, x_initializer, a_initializer, c_initializer)
    

def main():


    run_nn()


if __name__== "__main__":
  main()
