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

def generate_seq(
        model : Model,
        seed,
        numchars,
        size,
        temperature=1.0):
    """
    :param model: The complete RNN language model
    :param seed: The first few wordas of the sequence to start generating from
    :param size: The total size of the sequence to generate
    :param temperature: This controls how much we follow the probabilities provided by the network. For t=1.0 we just
        sample directly according to the probabilities. Lower temperatures make the high-probability words more likely
        (providing more likely, but slightly boring sentences) and higher temperatures make the lower probabilities more
        likely (resulting is weirder sentences). For temperature=0.0, the generation is _greedy_, i.e. the word with the
        highest probability is always chosen.
    :return: A list of integers representing a samples sentence
    """
    ls = seed.shape[0]

    # Due to the way Keras RNNs work, we feed the model a complete sequence each time. At first it's just the seed,
    # zero-padded to the right length. With each iteration we sample and set the next character.

    tokens = np.concatenate([seed, np.zeros(size - ls)])

    # convert the integer sequence to a categorical one
    toh = util.to_categorical(tokens[None, :], numchars)

    for i in range(ls, size-1):

        # predict next characters (for the whole sequence)
        probs = model.predict(toh)

        # Extract the i-th probability vector and sample an index from it
        next_token = util.sample(probs[0, i-1, :], temperature)

        tokens[i] = next_token

        # update the one-hot encoding
        toh[0, i, 0] = 0
        toh[0, i, next_token] = 1

    return [int(t) for t in tokens]

def go(options):

    if options.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
        np.random.seed(seed)
    else:
        np.random.seed(options.seed)


    ## Load the data
    if options.task == 'alice':

        dir = options.data_dir
        x, char_to_ix, ix_to_char = \
            util.load_characters(util.DIR + '/datasets/alice.txt', limit=options.limit, length=options.sequence_length)

        x_max_len = max([len(sentence) for sentence in x])
        numchars = len(ix_to_char)
        print(numchars, ' distinct characters found')

        x = sequence.pad_sequences(x, x_max_len, padding='post', truncating='post')

    elif options.task == 'shakespeare':

        dir = options.data_dir
        x, char_to_ix, ix_to_char = \
            util.load_characters(util.DIR + '/datasets/shakespeare.txt', limit=options.limit, length=options.sequence_length)

        x_max_len = max([len(sentence) for sentence in x])
        numchars = len(ix_to_char)
        print(numchars, ' distinct characters found')

        x = sequence.pad_sequences(x, x_max_len, padding='post', truncating='post')

    elif options.task == 'file':

        dir = options.data_dir
        x, char_to_ix, ix_to_char = \
            util.load_characters(options.da, limit=options.limit, length=options.sequence_length)

        x_max_len = max([len(sentence) for sentence in x])
        numchars = len(ix_to_char)
        print(numchars, ' distinct characters found')

        x = sequence.pad_sequences(x, x_max_len, padding='post', truncating='post')

    else:
        raise Exception('Dataset name ({}) not recognized.'.format(options.task))

    def decode(seq):
        return ''.join(ix_to_char[id] for id in seq)

    print('Data Loaded.')

    ## Shape the data. The inputs get a start symbol (1) prepended. We shorten the sequences by one so that the lengths
    #  match
    n = x.shape[0]

    x_in  = np.concatenate([np.ones((n, 1)), x[:, :-1]], axis=1)  # prepend start symbol
    x_out = x
    assert x_in.shape == x_out.shape

    #  convert from integer sequences to sequences of one-hot vectors
    x_in = util.to_categorical(x_in, numchars)
    x_out = util.to_categorical(x_out, numchars)  # output to one-hots

    ## Define the model

    input = Input(shape=(None, numchars))
    #- We define the model as variable-length (even though all training data has fixed length). This allows us to generate
    #  longer sequences during inference.

    h = LSTM(options.lstm_capacity, return_sequences=True)(input)

    if options.extra is not None:
        for _ in range(options.extra):
            h = LSTM(options.lstm_capacity, return_sequences=True)(h)

    #  Apply a single dense layer to all timesteps of the resulting sequence to convert back to characters
    out = TimeDistributed(Dense(numchars, activation='softmax'))(h)

    model = Model(input, out)

    opt = keras.optimizers.Adam(lr=options.lr)

    model.compile(opt, 'categorical_crossentropy')
    #- For each timestep the model outputs a probability distribution over all characters. Categorical crossentopy mean
    #  that we try to optimize the log-probability of the probability of the correct character (averaged over all
    #  characters in all sequences.

    model.summary()

    ## Create callback to generate some samples after each epoch

    def generate(epoch):
        if epoch % options.out_every == 0 and epoch > 0:
            for i in range(CHECK):
                b = random.randint(0, n - 1)

                seed = x[b, :20]
                seed = np.insert(seed, 0, 1)
                gen = generate_seq(model, seed, numchars, options.gen_length)

                print('*** [', decode(seed), '] ', decode(gen[len(seed):]))
                print()

    # Train the model
    generate_stuff = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: generate(epoch))

    model.fit(x_in, x_out, epochs=options.epochs, batch_size=options.batch, callbacks=[generate_stuff])

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
                        default=5, type=int)

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
                        help="Task. Either 'shakespeare', 'alice' or 'file' (a custom text file specified with -D).",
                        default='alice', type=str)

    parser.add_argument("-D", "--data",
                        dest="data_dir",
                        help="Data file. Make sure to use '-t file'.",
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
                        default=100, type=int)

    parser.add_argument("-I", "--limit",
                        dest="limit",
                        help="Character cap for the corpus",
                        default=None, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random (seed is printed for reproducability).)",
                        default=-1, type=int)

    parser.add_argument("-x", "--extra-layers",
                        dest="extra",
                        help="Number of _extra_ LSTM layers (if None/0 the model will have 1 layer.",
                        default=2, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)