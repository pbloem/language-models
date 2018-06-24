import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import  LSTM, Embedding, TimeDistributed, Input, Dense
from keras.models import Model
from tensorflow.python.client import device_lib

from tqdm import tqdm
import os, random

from argparse import ArgumentParser

import numpy as np

from tensorboardX import SummaryWriter

import util

CHECK = 5

def generate_seq(model : Model, seed, size, temperature=1.0):
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

    for i in range(ls, size):

        probs = model.predict(tokens[None,:])

        # Extract the i-th probability vector and sample an index from it
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature)

        tokens[i] = next_token

    return [int(t) for t in tokens]

def sparse_loss(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def go(options):

    tbw = SummaryWriter(log_dir=options.tb_dir)

    if options.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
        np.random.seed(seed)
    else:
        np.random.seed(options.seed)

    if options.task == 'wikisimple':

        x, w21, i2w = \
            util.load_words(util.DIR + '/datasets/wikisimple.txt', vocab_size=options.top_words, limit=options.limit)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        numwords = len(i2w)
        print('max sequence length ', x_max_len)
        print(numwords, 'distinct words')

        x = util.batch_pad(x, options.batch, add_eos=True)

    elif options.task == 'file':

        x, w21, i2w = \
            util.load_words(options.data_dir, vocab_size=options.top_words, limit=options.limit)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        numwords = len(i2w)
        print('max sequence length ', x_max_len)
        print(numwords, 'distinct words')

        x = util.batch_pad(x, options.batch, add_eos=True)

    else:
        raise Exception('Task {} not recognized.'.format(options.task))

    def decode(seq):
        return ' '.join(i2w[id] for id in seq)

    print('Finished data loading. ', sum([b.shape[0] for b in x]), ' sentences loaded')

    ## Define model

    input = Input(shape=(None, ))
    embedding = Embedding(numwords, options.lstm_capacity, input_length=None)

    embedded = embedding(input)

    decoder_lstm = LSTM(options.lstm_capacity, return_sequences=True)
    h = decoder_lstm(embedded)

    if options.extra is not None:
        for _ in range(options.extra):
            h = LSTM(options.lstm_capacity, return_sequences=True)(h)

    fromhidden = Dense(numwords, activation='linear')
    out = TimeDistributed(fromhidden)(h)

    model = Model(input, out)

    opt = keras.optimizers.Adam(lr=options.lr)
    lss = sparse_loss

    model.compile(opt, lss)
    model.summary()

    ## Training

    #- Since we have a variable batch size, we make our own training loop, and train with
    #  model.train_on_batch(...). It's a little more verbose, but it gives us more control.

    epoch = 0
    instances_seen = 0
    while epoch < options.epochs:

        for batch in tqdm(x):
            n, l = batch.shape

            batch_shifted = np.concatenate([np.ones((n, 1)), batch], axis=1)  # prepend start symbol
            batch_out = np.concatenate([batch, np.zeros((n, 1))], axis=1)     # append pad symbol

            loss = model.train_on_batch(batch_shifted, batch_out[:, :, None])

            instances_seen += n
            tbw.add_scalar('lm/batch-loss', float(loss), instances_seen)

        epoch += 1

        # Show samples for some sentences from random batches
        for temp in [0.0, 0.9, 1, 1.1, 1.2]:
            print('### TEMP ', temp)
            for i in range(CHECK):
                b = random.choice(x)

                if b.shape[1] > 20:
                    seed = b[0,:20]
                else:
                    seed = b[0, :]

                seed = np.insert(seed, 0, 1)
                gen = generate_seq(model, seed,  60, temperature=temp)

                print('*** [', decode(seed), '] ', decode(gen[len(seed):]))

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=20, type=int)

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
                        default=128, type=int)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='wikisimple', type=str)

    parser.add_argument("-D", "--data-directory",
                        dest="data",
                        help="Data file. Should contain one sentence per line.",
                        default='./data', type=str)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)

    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words",
                        default=10000, type=int)

    parser.add_argument("-I", "--limit",
                        dest="limit",
                        help="Character cap for the corpus",
                        default=None, type=int)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/words', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random (seed is printed for reproducability).",
                        default=-1, type=int)

    parser.add_argument("-x", "--extra-layers",
                        dest="extra",
                        help="Number of extra LSTM layers.",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)