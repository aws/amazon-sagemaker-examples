import chainer
import chainer.functions as F
import chainer.links as L
import numpy
from chainer import reporter

embed_init = chainer.initializers.Uniform(0.25)


def sequence_embed(embed, xs, dropout=0.0):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.0):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class TextClassifier(chainer.Chain):

    """A classifier using a given encoder.

    This chain encodes a sentence and classifies it into classes.

    Args:
        encoder (Link): A callable encoder, which extracts a feature.
            Input is a list of variables whose shapes are
            "(sentence_length, )".
            Output is a variable whose shape is "(batchsize, n_units)".
        n_class (int): The number of classes to be predicted.

    """

    def __init__(self, encoder, n_class, dropout=0.1):
        super(TextClassifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, n_class)
        self.dropout = dropout

    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({"loss": loss.data}, self)
        reporter.report({"accuracy": accuracy.data}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        concat_encodings = F.dropout(self.encoder(xs), ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs


class RNNEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(RNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=embed_init)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)

        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert last_h.shape == (self.n_layers, len(xs), self.out_units)
        concat_outputs = last_h[-1]
        return concat_outputs


class CNNEncoder(chainer.Chain):

    """A CNN encoder with word embedding.

    This model encodes a sentence as a set of n-gram chunks
    using convolutional filters.
    Following the convolution, max-pooling is applied over time.
    Finally, the output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        out_units = n_units // 3
        super(CNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1, initialW=embed_init)
            self.cnn_w3 = L.Convolution2D(
                n_units, out_units, ksize=(3, 1), stride=1, pad=(2, 0), nobias=True
            )
            self.cnn_w4 = L.Convolution2D(
                n_units, out_units, ksize=(4, 1), stride=1, pad=(3, 0), nobias=True
            )
            self.cnn_w5 = L.Convolution2D(
                n_units, out_units, ksize=(5, 1), stride=1, pad=(4, 0), nobias=True
            )
            self.mlp = MLP(n_layers, out_units * 3, dropout)

        self.out_units = out_units * 3
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h_w5 = F.max(self.cnn_w5(ex_block), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        h = self.mlp(h)
        return h


class MLP(chainer.ChainList):

    """A multilayer perceptron.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units in a hidden or output layer.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_units, dropout=0.1):
        super(MLP, self).__init__()
        for i in range(n_layers):
            self.add_link(L.Linear(None, n_units))
        self.dropout = dropout
        self.out_units = n_units

    def __call__(self, x):
        for i, link in enumerate(self.children()):
            x = F.dropout(x, ratio=self.dropout)
            x = F.relu(link(x))
        return x


class BOWEncoder(chainer.Chain):

    """A BoW encoder with word embedding.

    This model encodes a sentence as just a set of words by averaging.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_vocab, n_units, dropout=0.1):
        super(BOWEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1, initialW=embed_init)

        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], numpy.int32)[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len
        return h


class BOWMLPEncoder(chainer.Chain):

    """A BOW encoder with word embedding and MLP.

    This model encodes a sentence as just a set of words by averaging.
    Additionally, its output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BOWMLPEncoder, self).__init__()
        with self.init_scope():
            self.bow_encoder = BOWEncoder(n_vocab, n_units, dropout)
            self.mlp_encoder = MLP(n_layers, n_units, dropout)

        self.out_units = n_units

    def __call__(self, xs):
        h = self.bow_encoder(xs)
        h = self.mlp_encoder(h)
        return h
