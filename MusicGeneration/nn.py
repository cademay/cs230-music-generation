import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from utils import generate_dataset_iterator, save_decoding
import progressbar

# Network Parameters
num_input = 128*2 + 100
timesteps = 100
num_hidden = 50 # hidden layer num of features
beta = .003
epochs = 100
threshold = .5
filename = 'out'
prob_1 = 0.0087827912566791136


def initialize_nn():

    # tf Graph input
    X = tf.placeholder("float", [1, timesteps, num_input])
    Y = tf.placeholder("float", [1, timesteps, num_input])
    print X
    print Y

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_input]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_input]))
    }

    def RNN(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, timesteps, 1)
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # print len(x)
        # print x[0]
        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        return tf.expand_dims(tf.matmul(tf.squeeze(outputs), weights['out']) + biases['out'], 0)
    logits = RNN(X, weights, biases)
    prediction = tf.nn.sigmoid(logits)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        logits=logits, targets=Y, pos_weight=1/prob_1))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)
    # Evaluate model (with test logits, for dropout to be disabled)
    # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    return init, train_op, X, Y, loss_op, prediction



def run_nn():
    init, train_op, X, Y, loss_op, prediction = initialize_nn()
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        step = 0
        bar = progressbar.ProgressBar(redirect_stdout=True, max_value=progressbar.UnknownLength)
        for epoch in range(epochs):
            devError = 0
            step = 0
            for decoding in generate_dataset_iterator('dev'):
                step += 1
                bar.update(step)
                for subset in range(decoding.shape[0] / timesteps):
                    subDecoding = decoding[subset * timesteps:]
                    if subDecoding.shape[0] < timesteps:
                        continue
                    dec = subDecoding[:timesteps]
                    batch_x = np.array([np.copy(dec)])
                    batch_y = np.array([np.concatenate([np.zeros((1,num_input)), dec[1:]], axis=0)])
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                    loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})
                    devError = devError * (1-beta) + beta * float(loss[0])
                    # print "Step " + str(step) + '.' + str(subset) +  ", Minibatch Loss= " + "{}".format(loss)
            testError = []
            for decoding in generate_dataset_iterator('test'):
                step += 1
                bar.update(step)
                for subset in range(decoding.shape[0] / timesteps):
                    subDecoding = decoding[subset * timesteps:]
                    if subDecoding.shape[0] < timesteps:
                        continue
                    dec = subDecoding[:timesteps]
                    batch_x = np.array([np.copy(dec)])
                    batch_y = np.array([np.concatenate([np.zeros((1,num_input)), dec[1:]], axis=0)])
                    loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})
                    testError.append(float(loss[0]))
            testError = np.mean(testError)
            print 'Epoch {}: Dev Error = {}, Test Error = {}'.format(epoch, devError, testError)
            fake = None
            for decoding in generate_dataset_iterator('dev'):
                fake = decoding
                break
            decoding = fake[:timesteps]
            for i in range(timesteps):
                batch_x = np.array([np.copy(decoding)])
                batch_y = np.array([np.concatenate([np.zeros((1,num_input)), decoding[1:]], axis=0)])
                pred = sess.run(prediction, feed_dict={X: batch_x, Y: batch_y})
                newFrame = pred[0][-1]
                # print newFrame
                on = np.argmax(newFrame[:128])
                off = np.argmax(newFrame[128:2*128])
                shift = np.argmax(newFrame[2*128:])
                newFrame = np.zeros(2*128 + 100)
                newFrame[on] = 1
                newFrame[128+off] = 1
                newFrame[2*128 + shift] = 1
                # newFrame[newFrame > threshold] = 1
                # newFrame[newFrame <= threshold] = 0
                decoding = np.concatenate([decoding[1:], [newFrame]], axis=0)
            save_decoding(decoding, filename + str(epoch) + '.mid')


def main():
    run_nn()


if __name__== "__main__":
  main()
