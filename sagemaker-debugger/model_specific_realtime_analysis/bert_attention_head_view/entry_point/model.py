# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""BertForQA models."""

__all__ = ['BertForQA', 'BertForQALoss']

from mxnet.gluon import HybridBlock, loss, nn
from mxnet.gluon.loss import Loss


class BertForQA(HybridBlock):
    """Model for SQuAD task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for QA task.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert, prefix=None, params=None):
        super(BertForQA, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.span_classifier = nn.Dense(units=2, flatten=False)

    def __call__(self, inputs, token_types, valid_length=None):
        #pylint: disable=arguments-differ, dangerous-default-value
        """Generate the unnormalized score for the given the input sequences."""
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        valid_length = [] if valid_length is None else valid_length
        return super(BertForQA, self).__call__(inputs, token_types, valid_length)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, seq_length, 2)
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        if isinstance(valid_length, list) and len(valid_length) == 0:
            valid_length = None
        bert_output = self.bert(inputs, token_types, valid_length)[0]

        output = self.span_classifier(bert_output)
        return output


class BertForQALoss(Loss):
    """Loss for SQuAD task with BERT.

    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):  # pylint: disable=unused-argument
        super(BertForQALoss, self).__init__(
            weight=None, batch_axis=0, **kwargs)
        self.loss = loss.SoftmaxCELoss()

    def hybrid_forward(self, F, pred, label):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        pred : NDArray, shape (batch_size, seq_length, 2)
            BERTSquad forward output.
        label : list, length is 2, each shape is (batch_size,1)
            label[0] is the starting position of the answer,
            label[1] is the ending position of the answer.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size,)
        """
        pred = F.split(pred, axis=2, num_outputs=2)
        start_pred = pred[0].reshape((0, -3))
        start_label = label[0]
        end_pred = pred[1].reshape((0, -3))
        end_label = label[1]
        return (self.loss(start_pred, start_label) + self.loss(
            end_pred, end_label)) / 2
