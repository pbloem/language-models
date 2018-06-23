import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import \
    Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape, LSTM, Embedding, RepeatVector,\
    TimeDistributed, Bidirectional, Concatenate, Lambda, SpatialDropout1D, Softmax, GRU
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from tensorboardX import SummaryWriter

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

import util

INDEX_FROM = 3
CHECK = 5
NINTER = 10

def anneal(step, total, k = 1.0, anneal_function='linear'):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-total/2))))

        elif anneal_function == 'linear':
            return min(1, step/(total/2))

def generate_seq(
        model : Model, z,
        size = 60,
        lstm_layer = None,
        seed = np.ones(1), temperature=1.0, stop_at_eos=False):

    # Keras doesn't allow us to easily execute sequence models step by step so we just feed it a zero-sequence multiple
    # times. At step i, we sample a word w from the predictions at i, and set that as element i+1 in the sequence.

    ls = seed.shape[0]
    tokens = np.concatenate([seed, np.zeros(size - ls)])

    for i in range(ls, size):

        probs = model.predict([tokens[None,:], z])

        # Extract the i-th probability vector and sample an index from it
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature)

        tokens[i] = next_token

    result = [int(t) for t in tokens]

    if stop_at_eos:
        if 3 in result and result.index(3) != len(result) - 1:
            result = result[:result.index(3)+1]

    return result


def decode_imdb(seq):

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}

    return ' '.join(id_to_word[id] for id in seq)

def sparse_loss(y_true, y_pred):
    losses = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return K.sum(losses, axis=-1)

def go(options):

    slength = options.max_length
    lstm_hidden = options.lstm_capacity

    print('devices', device_lib.list_local_devices())

    tbw = SummaryWriter(log_dir=options.tb_dir)

    if options.task == 'file':

        dir = options.data_dir
        x, x_vocab_len, x_word_to_ix, x_ix_to_word = \
            util.load_sentences(options.data_dir, vocab_size=options.top_words)

        if options.clip_length is not None:
            x = [(sentence if len(sentence) < options.clip_length else sentence[:options.clip_length]) for sentence in x]

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        print('max sequence length ', x_max_len)
        print(len(x_ix_to_word), 'distinct words')

        x = util.batch_pad(x, options.batch, add_eos=True)

        def decode(seq):
            return ' '.join(x_ix_to_word[id] for id in seq)

    elif options.task == 'europarl':

        dir = options.data_dir
        x, x_vocab_len, x_word_to_ix, x_ix_to_word, _, _, _, _ = \
            util.load_data(dir+os.sep+'europarl-v8.fi-en.en', dir+os.sep+'europarl-v8.fi-en.fi', vocab_size=options.top_words)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        print('max sequence length ', x_max_len)
        print(len(x_ix_to_word), 'distinct words')

        x = util.batch_pad(x, options.batch)

        # Padding zeros to make all sequences have a same length with the longest one
        # x = sequence.pad_sequences(x, maxlen=slength, dtype='int32', padding='post', truncating='post')
        # y = sequence.pad_sequences(y, maxlen=slength, dtype='int32', padding='post', truncating='post')

        def decode(seq):
            print(seq)
            return ' '.join(x_ix_to_word[id] for id in seq)

    else:
        # Load only training sequences
        (x, _), _ = imdb.load_data(num_words=options.top_words)

        # rm start symbol
        x = [l[1:] for l in x]

        # x = sequence.pad_sequences(x, maxlen=slength+1, padding='post', truncating='post')
        # x = x[:, 1:] # rm start symbol

        x = util.batch_pad(x, options.batch)

        decode = decode_imdb

    print('Data Loaded.')

    print(sum([b.shape[0] for b in x]), ' sentences loaded')

    # for i in range(3):
    #     print(x[i, :])
    #     print(decode(x[i, :]))

    num_words = len(x_ix_to_word)

    ## Define encoder
    input = Input(shape=(None, ), name='inp')

    embedding = Embedding(num_words, options.embedding_size, input_length=None)
    embedded = embedding(input)

    encoder = LSTM(lstm_hidden) if options.rnn_type == 'lstm' else GRU(lstm_hidden)
    h = Bidirectional(encoder)(embedded)

    tozmean = Dense(options.hidden)
    zmean = tozmean(h)

    tozlsigma = Dense(options.hidden)
    zlsigma = tozlsigma(h)

    ## Define KL Loss and sampling

    kl = util.KLLayer(weight = K.variable(1.0)) # computes the KL loss and stores it for later
    zmean, zlsigma = kl([zmean, zlsigma])

    eps = Input(shape=(options.hidden,), name='inp-epsilon')

    sample = util.Sample()
    zsample = sample([zmean, zlsigma, eps])

    ## Define decoder
    # zsample = Input(shape=(options.hidden,), name='inp-decoder-z')
    input_shifted = Input(shape=(None, ), name='inp-shifted')

    if options.rnn_type == 'lstm':
        expandz_h = Dense(lstm_hidden, input_shape=(options.hidden,))
        expandz_c = Dense(lstm_hidden, input_shape=(options.hidden,))
        z_exp_h = expandz_h(zsample)
        z_exp_c = expandz_c(zsample)
        state = [z_exp_h, z_exp_c]
    else:
        expandz = Dense(lstm_hidden, input_shape=(options.hidden,))
        state = expandz(zsample)

    seq = embedding(input_shifted)
    seq = SpatialDropout1D(rate=options.dropout)(seq)

    decoder_rnn = LSTM(lstm_hidden, return_sequences=True) if options.rnn_type == 'lstm' else GRU(lstm_hidden, return_sequences=True)
    h = decoder_rnn(seq, initial_state=state)

    towords = TimeDistributed(Dense(num_words))
    out = towords(h)

    auto = Model([input, input_shifted, eps], out)

    ## Extract the encoder and decoder models form the autoencoder

    # - NB: This isn't exactly DRY. It seems much nicer to build a separate encoder and decoder model and then build a
    #   an autoencoder model that chains the two. For the life of me, I couldn't get it to work. For some reason the
    #   gradients don't seem to propagate down to the encoder. Let me know if you have better luck.

    encoder = Model(input, [zmean, zlsigma])

    z_in = Input(shape=(options.hidden,))
    s_in = Input(shape=(None,))
    seq = embedding(s_in)
    if options.rnn_type == 'lstm':
        z_exp_h = expandz_h(z_in)
        z_exp_c = expandz_c(z_in)
        state = [z_exp_h, z_exp_c]
    else:
        state = expandz(z_in)
    h = decoder_rnn(seq, initial_state=state)
    out = towords(h)
    decoder = Model([s_in, z_in], out)

    ## Compile the autoencoder model
    if options.num_gpu is not None:
        auto = multi_gpu_model(auto, gpus=options.num_gpu)

    opt = keras.optimizers.Adam(lr=options.lr)

    auto.compile(opt, sparse_loss)
    auto.summary()

    epoch = 0
    instances_seen = 0

    # DEBUG
    # x = x[:20]

    while epoch < options.epochs:

        klw = anneal(epoch, options.epochs)
        print('EPOCH {:03}: Set KL weight to {}'.format(epoch, klw))
        K.set_value(kl.weight, klw)

        for batch in tqdm(x):

            n, l = batch.shape

            batch_shifted = np.concatenate([np.ones((n, 1)), batch], axis=1)            # prepend start symbol
            batch_out = np.concatenate([batch, np.zeros((n, 1))], axis=1)[:, :, None]   # append pad symbol
            eps = np.random.randn(n, options.hidden)   # random noise for the sampling layer

            loss = auto.train_on_batch([batch, batch_shifted, eps], batch_out)

            instances_seen += n
            tbw.add_scalar('seq2seq/batch-loss', float(loss)/l, instances_seen)

        epoch += 1

        if epoch % options.out_every == 0:

            # show samples for some sentences from random batches
            for i in range(CHECK):

                # CHECK 1. Generate some sentences from a z vector for a random
                # sentence from the corpus
                b = random.choice(x)

                z, _ = encoder.predict(b)
                z = z[None, 0, :]

                print('in             ',  decode(b[0, :]))

                l = 60 if options.clip_length is None else options.clip_length

                gen = generate_seq(decoder, z=z, size=l)
                print('out 1          ', decode(gen))
                gen = generate_seq(decoder, z=z, size=l, temperature=0.05)
                print('out 2 (t0.05)  ', decode(gen))
                gen = generate_seq(decoder, z=z, size=l, temperature=0.0)
                print('out 3 (greedy) ', decode(gen))

                # CHECK 2. Show the argmax reconstruction (i
                n, _ = b.shape
                b_shifted = np.concatenate([np.ones((n, 1)), b], axis=1)  # prepend start symbol
                eps = np.random.randn(n, options.hidden)   # random noise for the sampling layer

                out = auto.predict([b, b_shifted, eps])[None, 0, :]
                out = np.argmax(out[0, ...], axis=1)
                print(out)
                print('recon ',  decode([int(o) for o in out]))

                print()

            for i in range(CHECK):

                # CHECK 3: Sample two z's from N(0,1) and interpolate between them
                # Here we use use greedy decoding: i.e. we pick the word that gets the highest
                # probability

                zfrom, zto = np.random.randn(1, options.hidden), np.random.randn(1, options.hidden)

                for d in np.linspace(0, 1, num=NINTER):
                    z = zfrom * (1-d) + zto * d
                    gen = generate_seq(decoder, z=z, size=l, temperature=0.0)
                    print('out (d={:.1}) \t'.format(d), decode(gen))
                print()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-R", "--rnn-type",
                        dest="rnn_type",
                        help="Type of RNN to use [lstm, gru].",
                        default='lstm', type=str)

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
                        default=0.0001, type=float)

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

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs', type=str)

    parser.add_argument("-d", "--dropout-rate",
                        dest="dropout",
                        help="The word dropout rate used when training the decoder",
                        default=0.5, type=float)

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
                        help="Latent vector size",
                        default=16, type=int)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)

    parser.add_argument("-g", "--num-gpu",
                        dest="num_gpu",
                        help="How many GPUs to use (Default is 1 if available, You only need to set this if you wish to use more than 1).",
                        default=None, type=int)

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)

    parser.add_argument("-C", "--clip_length",
                        dest="clip_length",
                        help="If not None, all sentences longer than this length are clipped to this length.",
                        default=None, type=int)

    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words",
                        default=10000, type=int)
    #
    # parser.add_argument("-S", "--use-state",
    #                     dest="use_state",
    #                     help="Use the last hidden (C) state of the encoder LSTM instead of the last output vector.",
    #                     action='store_true')

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)