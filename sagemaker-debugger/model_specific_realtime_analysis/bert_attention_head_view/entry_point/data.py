# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT for QA datasets."""
import collections
import multiprocessing as mp
import time
from functools import partial

from mxnet.gluon.data import SimpleDataset
from gluonnlp.data.utils import whitespace_splitter
import numpy as np

__all__ = ['SQuADTransform', '\rocess_dataset']

class SquadExample:
    """A single training/test example for SQuAD question.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 example_id,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.example_id = example_id

def _worker_fn(example, transform):
    """Function for processing data in worker process."""
    feature = transform(example)
    return feature


def preprocess_dataset(dataset, transform, num_workers=8):
    """Use multiprocessing to perform transform for dataset.

    Parameters
    ----------
    dataset: dataset-like object
        Source dataset.
    transform: callable
        Transformer function.
    num_workers: int, default 8
        The number of multiprocessing workers to use for data preprocessing.

    """
    worker_fn = partial(_worker_fn, transform=transform)
    start = time.time()

    pool = mp.Pool(num_workers)
    dataset_transform = []
    dataset_len = []

    for data in pool.map(worker_fn, dataset):
        if data:
            for _data in data:
                dataset_transform.append(_data[:-1])
                dataset_len.append(_data[-1])
    
    dataset = SimpleDataset(dataset_transform).transform(
        lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))

    end = time.time()
    pool.close()

    print('Done! Transform dataset costs %.2f seconds.' % (end-start))
    return dataset, dataset_len


class SQuADFeature:
    """Single feature of a single example transform of the SQuAD question.

    """

    def __init__(self,
                 example_id,
                 qas_id,
                 doc_tokens,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 valid_length,
                 segment_ids,
                 start_position,
                 end_position,
                 is_impossible):
        self.example_id = example_id
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.valid_length = valid_length
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SQuADTransform:
    """Dataset Transformation for BERT-style QA.

    The transformation is processed in the following steps:
    - Convert from gluonnlp.data.SQuAD's record to SquadExample.
    - Tokenize the question_text in the example.
    - For examples where the document is too long,
      use a sliding window to split into multiple features and
      record whether each token is a maximum context.
    - Tokenize the split document chunks.
    - Combine the token of question_text with the token
      of the document and insert [CLS] and [SEP].
    - Generate the start position and end position of the answer.
    - Generate valid length.

    E.g:

    Inputs:

        question_text: 'When did BBC Japan begin broadcasting?'
        doc_tokens: ['BBC','Japan','was','a','general','entertainment','channel,',
                    'which','operated','between','December','2004','and','April',
                    '2006.','It','ceased','operations','after','its','Japanese',
                    'distributor','folded.']
        start_position: 10
        end_position: 11
        orig_answer_text: 'December 2004'

    Processed:

        tokens: ['[CLS]','when','did','bbc','japan','begin','broadcasting','?',
                '[SEP]','bbc','japan','was','a','general','entertainment','channel',
                ',','which','operated','between','december','2004','and','april',
                '2006','.','it','ceased','operations','after','its','japanese',
                'distributor','folded','.','[SEP]']
        segment_ids: [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        start_position: 20
        end_position: 21
        valid_length: 36

    Because of the sliding window approach taken to scoring documents, a single
    token can appear in multiple documents.
    So you need to record whether each token is a maximum context. E.g.
       Doc: the man went to the store and bought a gallon of milk
       Span A: the man went to the
       Span B: to the store and bought
       Span C: and bought a gallon of
       ...

    Now the word 'bought' will have two scores from spans B and C. We only
    want to consider the score with "maximum context", which we define as
    the *minimum* of its left and right context (the *sum* of left and
    right context will always be the same, of course).

    In the example the maximum context for 'bought' would be span C since
    it has 1 left context and 3 right context, while span B has 4 left context
    and 0 right context.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    labels : list of int.
        List of all label ids for the classification task.
    max_seq_length : int, default 384
        Maximum sequence length of the sentences.
    doc_stride : int, default 128
        When splitting up a long document into chunks,
        how much stride to take between chunks.
    max_query_length : int, default 64
        The maximum length of the query tokens.
    is_pad : bool, default True
        Whether to pad the sentences to maximum length.
    is_training : bool, default True
        Whether to run training.
    do_lookup : bool, default True
        Whether to do vocabulary lookup for convert tokens to indices.
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length=384,
                 doc_stride=128,
                 max_query_length=64,
                 is_pad=True,
                 is_training=True,
                 do_lookup=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.is_pad = is_pad
        self.is_training = is_training
        self.do_lookup = do_lookup

    def _is_whitespace(self, c):
        if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(
                c) == 0x202F:
            return True
        return False

    def _toSquadExample(self, record):
        example_id = record[0]
        qas_id = record[1]
        question_text = record[2]
        paragraph_text = record[3]
        orig_answer_text = record[4][0] if record[4] else ''
        answer_offset = record[5][0] if record[5] else ''
        is_impossible = record[6] if len(record) == 7 else False

        doc_tokens = []

        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if self._is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        start_position = -1
        end_position = -1

        if self.is_training:
            if not is_impossible:
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[
                    answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = ' '.join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = ' '.join(
                    whitespace_splitter(orig_answer_text.strip()))
                if actual_text.find(cleaned_answer_text) == -1:
                    print('Could not find answer: %s vs. %s' %
                          (actual_text, cleaned_answer_text))
                    return None
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ''

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            example_id=example_id,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        return example

    def _transform(self, *record):
        example = self._toSquadExample(record)
       
        if not example:
            return None

        padding = self.tokenizer.vocab.padding_token
        if self.do_lookup:
            padding = self.tokenizer.vocab[padding]
        features = []
        query_tokens = self.tokenizer(example.question_text)

        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[0:self.max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if self.is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if self.is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position +
                                                     1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                self.tokenizer, example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            'DocSpan', ['start', 'length'])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append(self.tokenizer.vocab.cls_token)
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append(self.tokenizer.vocab.sep_token)
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append(self.tokenizer.vocab.sep_token)
            segment_ids.append(1)

            if self.do_lookup:
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                input_ids = tokens

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            valid_length = len(input_ids)

            # Zero-pad up to the sequence length.
            if self.is_pad:
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(padding)
                    segment_ids.append(padding)

                assert len(input_ids) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

            start_position = 0
            end_position = 0
            if self.is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if self.is_training and example.is_impossible:
                start_position = 0
                end_position = 0
       
            features.append(SQuADFeature(example_id=example.example_id,
                                         qas_id=example.qas_id,
                                         doc_tokens=example.doc_tokens,
                                         doc_span_index=doc_span_index,
                                         tokens=tokens,
                                         token_to_orig_map=token_to_orig_map,
                                         token_is_max_context=token_is_max_context,
                                         input_ids=input_ids,
                                         valid_length=valid_length,
                                         segment_ids=segment_ids,
                                         start_position=start_position,
                                         end_position=end_position,
                                         is_impossible=example.is_impossible))
        
        return features

    def __call__(self, record):
        examples = self._transform(*record)
        if not examples:
            return None
        features = []

        for _example in examples:
            feature = []
            feature.append(_example.example_id)
            feature.append(_example.input_ids)
            feature.append(_example.segment_ids)
            feature.append(_example.valid_length)
            feature.append(_example.start_position)
            feature.append(_example.end_position)  
            feature.append(len(_example.input_ids))
            features.append(feature)
     
        return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = ' '.join(tokenizer(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
