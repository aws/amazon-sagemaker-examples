# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
#    http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp
import scipy.stats

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def generate_griffiths_data(num_documents=5000, average_document_length=150,
                            num_topics=5, alpha=None, eta=None, seed=0):
    """Returns example documents from Griffiths-Steyvers [1].

    Given an `alpha` and `eta, the Dirichlet priors for the topic and topic-word
    distributions respectively, this function generates sample document word
    counts according to the Latent Dirichlet Allocation (LDA) model.

    Parameters
    ----------
    num_documents : int
        (Default: 1000) The number of example documents to create using LDA.
    average_document_length : int
        (Default: 100) The average number of words in each document. The
        document length is sampled from a Poisson distribution with this mean.
    num_topics : int
        (Default: 10) Can be set to either 5 or 10. The number of known topics.
    alpha : Numpy NDArray
        (Default: None) An array of length `num_topics` representing a given
        Dirichlet topic prior. If `None` is provided then a uniform
        distribution will be used.
    eta : Numpy NDArray
        (Default: None) An array of length `num_topics` representing a given
        Dirichlet topic-word prior.
    seed : int
        (Defualt: 0) The random number generator seed.

    Returns
    -------
    alpha : Numpy NDArray
        A vector of length `num_topics` equal to the Dirichlet prior used to
        generate documents.
    beta : Numpy NDArray
        A matrix of size `num_topics` x 25 equal to the topic-word probability
        matrix used to generate documents.
    documents : Numpy NDArray
        A matrix of size `num_documents` x 25 equal to the documents generated
        by the LDA model defined by `alpha` and `beta.
    theta : Numpy NDArray
        A matrix of size `num_documents` x `num_topics` equal to the topic
        mixtures used to generate the output `documents`.

    References
    ----------
    [1] Thomas L Griffiths and Mark Steyvers. "Finding Scientific Topics."
        Proceedings of the National Academy of Sciences, 101(suppl 1):5228–5235,
        2004.

    """
    vocabulary_size = 25
    image_dim = np.int(np.sqrt(vocabulary_size))

    # perform checks on input
    assert num_topics in [5,10], 'Example data only available for 5 or 10 topics'
    if alpha:
        assert len(alpha) == num_topics, 'len(alpha) must be equal to num_topics'

    # initialize Dirichlet alpha and eta distributions if not provided. here,
    # the eta distribution is only across `image_dim` elements since each
    # topic-word distribution will only have `image_dim` non-zero entries
    #
    np.random.seed(seed=seed)
    if alpha is None:
        alpha = np.ones(num_topics, dtype=np.float) / num_topics
    if eta is None:
        eta = [100]*image_dim  # make it close to a uniform distribution
    dirichlet_alpha = sp.stats.dirichlet(alpha)
    dirichlet_eta = sp.stats.dirichlet(eta)

    # initialize a known topic-word distribution (beta) using eta. these are
    # the "row" and "column" topics, respectively. when num_topics = 5 only
    # create the col topics. when num_topics = 10 add the row topics as well
    #
    beta = np.zeros((num_topics,image_dim,image_dim), dtype=np.float)
    for i in range(image_dim):
        beta[i,:,i] = dirichlet_eta.rvs(size=1)
    if num_topics == 10:
        for i in range(image_dim):
            beta[i+image_dim,i,:] = dirichlet_eta.rvs(size=1)
    beta.resize(num_topics, vocabulary_size)

    # generate documents using the LDA model / provess
    #
    document_lengths = sp.stats.poisson(average_document_length).rvs(size=num_documents)
    documents = np.zeros((num_documents,vocabulary_size), dtype=np.float)
    thetas = dirichlet_alpha.rvs(size=num_documents)  # precompute topic distributions for performance
    for m in range(num_documents):
        document_length = document_lengths[m]
        theta = thetas[m]
        topic = sp.stats.multinomial.rvs(1, theta, size=document_length)  # precompute topics for performance

        # generate word counts within document
        for n in range(document_length):
            word_topic = topic[n]
            topic_index = np.argmax(word_topic)
            topic_word_distribution = beta[topic_index]
            word = sp.stats.multinomial.rvs(1, topic_word_distribution, size=1).reshape(vocabulary_size)
            documents[m] += word

    return alpha, beta, documents, thetas

def plot_lda(data, nrows, ncols, with_colorbar=True, cmap=cm.viridis):
    """Helper function for plotting arrays of image"""
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols,nrows))
    vmin = 0
    vmax = data.max()

    V = len(data[0])
    n = int(np.sqrt(V))
    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols + j

            if nrows > 1:
                im = ax[i,j].matshow(data[index].reshape(n,n), cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax[j].matshow(data[index].reshape(n,n), cmap=cmap, vmin=vmin, vmax=vmax)

    for axi in ax.ravel():
        axi.set_xticks([])
        axi.set_yticks([])

    if with_colorbar:
        fig.colorbar(im, ax=ax.ravel().tolist(), orientation='horizontal', fraction=0.2)
    return fig

def match_estimated_topics(topics_known, topics_estimated):
    """A dumb but fast way to match known topics to estimated topics"""
    K, V = topics_known.shape
    permutation = -1*np.ones(K, dtype=np.int)
    unmatched_estimated_topics = []

    for estimated_topic_index, t in enumerate(topics_estimated):
        matched_known_topic_index = np.argmin([np.linalg.norm(known_topic - t) for known_topic in topics_known])
        if permutation[matched_known_topic_index] == -1:
            permutation[matched_known_topic_index] = estimated_topic_index
        else:
            unmatched_estimated_topics.append(estimated_topic_index)

    for estimated_topic_index in unmatched_estimated_topics:
        for i in range(K):
            if permutation[i] == -1:
                permutation[i] = estimated_topic_index
                break

    return permutation, (topics_estimated[permutation,:]).copy()

def _document_with_topic(fig, gsi, index, document, topic_mixture=None,
                         vmin=0, vmax=32):
    ax_doc = fig.add_subplot(gsi[:5,:])
    ax_doc.matshow(document.reshape(5,5), cmap='gray_r',
                   vmin=vmin, vmax=vmax)
    ax_doc.set_xticks([])
    ax_doc.set_yticks([])

    if topic_mixture is not None:
        ax_topic = plt.subplot(gsi[-1,:])
        ax_topic.matshow(topic_mixture.reshape(1,-1), cmap='Reds',
                         vmin=0, vmax=1)
        ax_topic.set_xticks([])
        ax_topic.set_yticks([])

def plot_lda_topics(documents, nrows, ncols, with_colorbar=True,
                    topic_mixtures=None, cmap='Viridis', dpi=160):
    fig = plt.figure()
    gs = GridSpec(nrows, ncols)

    vmin, vmax = (0, documents.max())

    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols + j
            gsi = GridSpecFromSubplotSpec(6, 5, subplot_spec=gs[i,j])
            _document_with_topic(fig, gsi, index, documents[index],
                                 topic_mixture=topic_mixtures[index],
                                 vmin=vmin, vmax=vmax)

    return fig
