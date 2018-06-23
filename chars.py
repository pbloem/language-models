import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.layers import LSTM, TimeDistributed, Input, Dense
from keras.models import Model

import os, random
import numpy as np

from argparse import ArgumentParser


import util

INDEX_FROM = 3
CHECK = 5

def sample(preds, temperature=1.):
    """
    Sample an index from a probability vector.
    (Code copied from Keras docs)

    :param preds:
    :param temperature:
    :return:
    """

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)

def generate_seq(
        model : Model,
        seed,
        numchars,
        size):

    ls = seed.shape[0]

    # Due to the way Keras RNNs work, we feed the model the whole sequence each time. At first it's just the seed,
    # padded to the right length. With each iteration we sample and set the next character.

    tokens = np.concatenate([seed, np.zeros(size - ls)])

    for i in range(ls, size-1):

        # convert the integer sequence to a categorical one
        toh = util.to_categorical(tokens[None,:], numchars)
        # predict next characters (for the whole sequence)
        probs = model.predict(toh)

        # Extract the i-th probability vector and sample an index from it
        next_token = sample(probs[0, i, :])

        tokens[i+1] = next_token

    return [int(t) for t in tokens]

def go(options):

    if options.task == 'shakespeare':

        dir = options.data_dir
        x, char_to_ix, ix_to_char = \
            util.load_char_data('./datasets/shakespeare.txt', limit=options.limit, length=options.sequence_length)

        x_max_len = max([len(sentence) for sentence in x])
        numchars = len(ix_to_char)
        print(numchars, ' distinct characters found')

        x = sequence.pad_sequences(x, x_max_len, padding='post', truncating='post')

        def decode(seq):
            return ''.join(ix_to_char[id] for id in seq)

    if options.task == 'file':

        dir = options.data_dir
        x, char_to_ix, ix_to_char = \
            util.load_char_data(options.da, limit=options.limit, length=options.sequence_length)

        x_max_len = max([len(sentence) for sentence in x])
        numchars = len(ix_to_char)
        print(numchars, ' distinct characters found')

        x = sequence.pad_sequences(x, x_max_len, padding='post', truncating='post')

        def decode(seq):
            return ''.join(ix_to_char[id] for id in seq)

    else:
        raise Exception('Dataset name not recognized.')

    print('Data Loaded.')

    ## Define model
    input = Input(shape=(None, numchars))

    h = LSTM(options.lstm_capacity, return_sequences=True)(input)

    if options.extra is not None:
        for _ in range(options.extra):
            h = LSTM(options.lstm_capacity, return_sequences=True)(h)

    out = TimeDistributed(Dense(numchars, activation='softmax'))(h)

    model = Model(input, out)

    opt = keras.optimizers.Adam(lr=options.lr)

    model.compile(opt, 'categorical_crossentropy')
    model.summary()

    n = x.shape[0]

    x_shifted = np.concatenate([np.ones((n, 1)), x], axis=1)  # prepend start symbol
    x_shifted = util.to_categorical(x_shifted, numchars)

    x_out = np.concatenate([x, np.zeros((n, 1))], axis=1)  # append pad symbol
    x_out = util.to_categorical(x_out, numchars)  # output to one-hots

    ## Create callback to generate some sample after each epoch
    def generate():
        for i in range(CHECK):
            b = random.randint(0, n - 1)

            seed = x[b, :20]
            seed = np.insert(seed, 0, 1)
            gen = generate_seq(model, seed, numchars, options.gen_length)

            print('*** [', decode(seed), '] ', decode(gen[len(seed):]))
            print()

    # Train the model
    generate_stuff = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: generate())

    model.fit(x_shifted, x_out, epochs=options.epochs, batch_size=64, callbacks=[generate_stuff])

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the word embeddings on the input layer.",
                        default=300, type=int)

    parser.add_argument("-o", "--output-every",
                        dest="out_every",
                        help="Output every n epochs.",
                        default=1, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='shakespeare', type=str)

    parser.add_argument("-D", "--data",
                        dest="data_dir",
                        help="Data file",
                        default=None, type=str)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)

    parser.add_argument("-m", "--sequence_length",
                        dest="sequence_length",
                        help="Sequence length",
                        default=None, type=int)


    parser.add_argument("-g", "--gen_length",
                        dest="gen_length",
                        help="How many characted to generate for each sample",
                        default=400, type=int)

    parser.add_argument("-I", "--limit",
                        dest="limit",
                        help="Character cap for the corpus",
                        default=None, type=int)

    parser.add_argument("-x", "--extra-layers",
                        dest="extra",
                        help="Number of _extra_ LSTM layers (if None/0 the model will have 1 layer.",
                        default=2, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)