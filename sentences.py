import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import \
    Dense, LSTM, Embedding, TimeDistributed, Bidirectional, SpatialDropout1D, GRU, Input
from keras.models import Model
from tensorflow.python.client import device_lib

from tensorboardX import SummaryWriter

from keras.utils import multi_gpu_model

from tqdm import tqdm
import math, sys, os, random
import numpy as np

from argparse import ArgumentParser

import util

CHECK = 5
NINTER = 10

def anneal(step, total, k = 1.0, anneal_function='linear'):
    """
    Compute the annealing schedule for the KL weight.
    :param step: The current epoch
    :param total:  The total nr. of epochs
    :param k: Scaling function for the logistic schedule
    :param anneal_function: Logistic or linear.
    :return:
    """
    if anneal_function == 'logistic':
       return float(1/(1+np.exp(-k*(step-total/2))))

    elif anneal_function == 'linear':
       return min(1, step/(total/2))

def generate_seq(
        model : Model, z,
        size = 60,
        seed = np.ones(1), temperature=1.0, stop_at_eos=True):
    """

    :param model:
    :param z: The latent vector from which to generate
    :param size:
    :param lstm_layer:
    :param seed:
    :param temperature: This controls how much we follow the probabilities provided by the network. For t=1.0 we just
        sample directly according to the probabilities. Lower temperatures make the high-probability words more likely
        (providing more likely, but slightly boring sentences) and higher temperatures make the lower probabilities more
        likely (resulting is weirder sentences). For temperature=0.0, the generation is _greedy_, i.e. the word with the
        highest probability is always chosen.
    :param stop_at_eos: If true anything after the first end-of-sentence symbol is ignored.
    :return: A list of integers representing a sentence.
    """

    # Due to the way Keras RNNs work, we feed the model a complete sequence each time. At first it's just the seed,
    # zero-padded to the right length. With each iteration we sample and set the next character.

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

def sparse_loss(y_true, y_pred):
    losses = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return K.sum(losses, axis=-1) # Note the sum over timesteps. This is crucial for the VAE

def go(options):

    tbw = SummaryWriter(log_dir=options.tb_dir)

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

    ## Define encoder
    input = Input(shape=(None, ), name='inp')

    embedding = Embedding(numwords, options.embedding_size, input_length=None)
    embedded = embedding(input)

    encoder = LSTM(options.lstm_capacity) if options.rnn_type == 'lstm' else GRU(options.lstm_capacity)
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
    input_shifted = Input(shape=(None, ), name='inp-shifted')

    expandz_h = Dense(options.lstm_capacity, input_shape=(options.hidden,))
    expandz_c = Dense(options.lstm_capacity, input_shape=(options.hidden,))
    z_exp_h = expandz_h(zsample)
    z_exp_c = expandz_c(zsample)
    state = [z_exp_h, z_exp_c]


    seq = embedding(input_shifted)
    seq = SpatialDropout1D(rate=options.dropout)(seq)

    decoder_rnn = LSTM(options.lstm_capacity, return_sequences=True)
    h = decoder_rnn(seq, initial_state=state)

    towords = TimeDistributed(Dense(numwords))
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
    z_exp_h = expandz_h(z_in)
    z_exp_c = expandz_c(z_in)
    state = [z_exp_h, z_exp_c]
    h = decoder_rnn(seq, initial_state=state)
    out = towords(h)
    decoder = Model([s_in, z_in], out)

    ## Compile the autoencoder model for training
    opt = keras.optimizers.Adam(lr=options.lr)

    auto.compile(opt, sparse_loss)
    auto.summary()

    instances_seen = 0
    for epoch in range(options.epochs+1):

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


        if epoch % options.out_every == 0 and epoch > 0:

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
                        default='wikisimple', type=str)

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

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)

    parser.add_argument("-C", "--clip_length",
                        dest="clip_length",
                        help="If not None, all sentences longer than this length are clipped to this length.",
                        default=None, type=int)

    parser.add_argument("-I", "--limit",
                        dest="limit",
                        help="Character cap for the corpus",
                        default=None, type=int)

    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words",
                        default=10000, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)