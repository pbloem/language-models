import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import \
    Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape, LSTM, Embedding, RepeatVector,\
    TimeDistributed, Bidirectional, Concatenate, Lambda, SpatialDropout1D, Softmax
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from keras.utils import multi_gpu_model

import tensorflow as tf

from sklearn import datasets

from tqdm import tqdm
import math, sys, os, random
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential
from keras.engine.topology import Layer
from keras.utils import to_categorical

import numpy as np

from tensorboardX import SummaryWriter

import util

INDEX_FROM = 3
CHECK = 5



def generate_seq(model : Model, seed, size, temperature=1.0):

    ls = seed.shape[0]

    # Due to the way Keras RNNs work, we feed the model the
    # whole sequence each time, constantly sampling the nect word.
    # It's a little bit inefficient, but that doesn't matter much when generating

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
    slength = options.max_length
    top_words = options.top_words
    lstm_hidden = options.lstm_capacity

    print('devices', device_lib.list_local_devices())

    tbw = SummaryWriter(log_dir=options.tb_dir)

    if options.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
        np.random.seed(seed)
    else:
        np.random.seed(options.seed)

    if options.task == 'file':

        dir = options.data_dir
        x, x_vocab_len, x_word_to_ix, x_ix_to_word = \
            util.load_sentences(options.data_dir, vocab_size=top_words, limit=options.limit)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        print('max sequence length ', x_max_len)
        print(len(x_ix_to_word), 'distinct words')

        x = util.batch_pad(x, options.batch)

        def decode(seq):
            return ' '.join(x_ix_to_word[id] for id in seq)

    elif options.task == 'europarl':

        dir = options.data_dir
        x, x_vocab_len, x_word_to_ix, x_ix_to_word, _, _, _, _ = \
            util.load_data(dir+os.sep+'europarl-v8.fi-en.en', dir+os.sep+'europarl-v8.fi-en.fi', vocab_size=top_words, limit=options.limit)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        print('max sequence length ', x_max_len)
        print(len(x_ix_to_word), 'distinct words')

        x = util.batch_pad(x, options.batch)

        def decode(seq):
            return ' '.join(x_ix_to_word[id] for id in seq)

    elif options.task == 'imdb':
        # Load only training sequences
        (x, _), _ = imdb.load_data(num_words=top_words)

        # rm start symbol
        x = [l[1:] for l in x]

        # x = sequence.pad_sequences(x, maxlen=slength+1, padding='post', truncating='post')
        # x = x[:, 1:] # rm start symbol

        x = util.batch_pad(x, options.batch)

        word_to_id = keras.datasets.imdb.get_word_index()
        word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        word_to_id["???"] = 3

        id_to_word = {value: key for key, value in word_to_id.items()}

        def decode(seq):
            return ' '.join(id_to_word[id] for id in seq)
    else:
        raise Exception('Task {} not recognized.'.format(options.task))

    print('Data Loaded.')

    print(sum([b.shape[0] for b in x]), ' sentences loaded')

    # for i in range(3):
    #     print(x[i, :])
    #     print(decode(x[i, :]))

    ## Define model
    input = Input(shape=(None, ))
    embedding = Embedding(top_words, lstm_hidden, input_length=None)

    embedded = embedding(input)

    decoder_lstm = LSTM(lstm_hidden, return_sequences=True)
    h = decoder_lstm(embedded)

    if options.extra is not None:
        for _ in range(options.extra):
            h = LSTM(lstm_hidden, return_sequences=True)(h)

    fromhidden = Dense(top_words, activation='linear')
    out = TimeDistributed(fromhidden)(h)

    model = Model(input, out)

    opt = keras.optimizers.Adam(lr=options.lr)
    lss = sparse_loss

    model.compile(opt, lss)
    model.summary()

    epochs = 0
    instances_seen = 0
    while epochs < options.epochs:

        for batch in tqdm(x):
            n, l = batch.shape

            batch_shifted = np.concatenate([np.ones((n, 1)), batch], axis=1)  # prepend start symbol
            batch_out = np.concatenate([batch, np.zeros((n, 1))], axis=1)     # append pad symbol

            loss = model.train_on_batch(batch_shifted, batch_out[:, :, None])

            instances_seen += n
            tbw.add_scalar('lm/batch-loss', float(loss), instances_seen)

        epochs += options.out_every

        # Show samples for some sentences from random batches
        for temp in [0.0, 0.7, 1, 1.3, 1.5]:
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
                        default=0.00001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='imdb', type=str)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
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
                        default='./runs/lm', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("-x", "--extra-layers",
                        dest="extra",
                        help="Number of extra LSTM layers",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)