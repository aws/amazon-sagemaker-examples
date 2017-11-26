# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
#    http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import io
from itertools import chain, islice
import json
import logging
import os
import pickle
import struct
import multiprocessing

from record_pb2 import Record
import boto3
from typing import Dict, Iterable, Mapping, Generator
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constants
JSON_SUFFIX = ".json"
ARG_SEPARATOR = ":"
BOS_SYMBOL = "<s>"
EOS_SYMBOL = "</s>"
UNK_SYMBOL = "<unk>"
PAD_SYMBOL = "<pad>"
PAD_ID = 0
TOKEN_SEPARATOR = " "
VOCAB_SYMBOLS = [PAD_SYMBOL, UNK_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
VOCAB_ENCODING = "utf-8"


# RecordIO and Protobuf related utilities

def write_recordio(f, data):
    kmagic = 0xced7230a
    length = len(data)
    f.write(struct.pack('I', kmagic))
    f.write(struct.pack('I', length))
    upper_align = ((length + 3) >> 2) << 2
    padding = bytes([0x00 for _ in range(upper_align - length)])
    f.write(data)
    f.write(padding)


def list_to_record_bytes(source: List[int] = None, target: List[int] = None):
    record = Record()
    record.features['source'].int32_tensor.values.extend(source)
    record.features['target'].int32_tensor.values.extend(target)
    return record.SerializeToString()


def read_next(f):
    kmagic = 0xced7230a
    raw_bytes = f.read(4)
    if not raw_bytes:
        return
    m = struct.unpack('I', raw_bytes)[0]
    if m != kmagic:
        raise ValueError("Incorrect encoding")
    length = struct.unpack('I', f.read(4))[0]
    upper_align = ((length + 3) >> 2) << 2
    data = f.read(upper_align)
    return data[:length]


def to_proto(f, sources, targets):
    for source, target in zip(sources, targets):
        record = list_to_record_bytes(source, target)
        write_recordio(f, record)


def write_to_s3(fobj, bucket, key):
    return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)


def upload_to_s3(bucket, key, sources, targets):
    f = io.BytesIO()
    to_proto(f, sources, targets)
    f.seek(0)
    url = 's3n://{}/{}'.format(bucket, key)
    print('Writing to {}'.format(url))
    write_to_s3(f, bucket, key)
    print('Done writing to {}'.format(url))


def smart_open(filename: str, mode: str = "rt", ftype: str = "auto", errors: str = 'replace'):
    """
    Returns a file descriptor for filename with UTF-8 encoding.
    If mode is "rt", file is opened read-only.
    If ftype is "auto", uses gzip iff filename endswith .gz.
    If ftype is {"gzip","gz"}, uses gzip.

    Note: encoding error handling defaults to "replace"

    :param filename: The filename to open.
    :param mode: Reader mode.
    :param ftype: File type. If 'auto' checks filename suffix for gz to try gzip.open
    :param errors: Encoding error handling during reading. Defaults to 'replace'
    :return: File descriptor
    """
    if ftype == 'gzip' or ftype == 'gz' or (ftype == 'auto' and filename.endswith(".gz")):
        return gzip.open(filename, mode=mode, encoding='utf-8', errors=errors)
    else:
        return open(filename, mode=mode, encoding='utf-8', errors=errors)


def get_tokens(line: str) -> Generator[str, None, None]:
    """
    Yields tokens from input string.

    :param line: Input string.
    :return: Iterator over tokens.
    """
    for token in line.rstrip().split():
        if len(token) > 0:
            yield token


def add_optional_args(model_params):
    model_params.add_argument('-vs', '--val-source', required=False, type=str, help='Validation source file.')
    model_params.add_argument('-vt', '--val-target', required=False, type=str, help='Validation target file.')
    model_params.add_argument('-to', '--train-output', required=False, type=str, default="train.rec",
                              help="Output filename (protobuf encoded .rec file) to write the processed train file. "
                                   "Default: %(default)s")
    model_params.add_argument('-vo', '--val-output', required=False, type=str, default="val.rec",
                              help="Output filename (protobuf encoded .rec file) to write the processed validation "
                                   "file. Default: %(default)s")
    model_params.add_argument('-single-vocab', action="store_true", default=False,
                              help="Include this flag to build a single vocab for both source and target.")
    model_params.add_argument('--vocab-source-json', required=False, type=str, default=None,
                              help="Path to source vocab json if it already exists")
    model_params.add_argument('--vocab-target-json', required=False, type=str, default=None,
                              help="Path to vocab target json if it already exists")
    model_params.add_argument('--num-words-source', required=False, type=int, default=50000,
                              help='Maximum vocabulary size for source. Default: %(default)s')
    model_params.add_argument('--num-words-target', required=False, type=int, default=50000,
                              help='Maximum vocabulary size for target. Default: %(default)s')
    model_params.add_argument('--word-min-count-source', required=False, type=int, default=1,
                              help='Minimum frequency of words to be included in source vocabulary. '
                                   'Default: %(default)s')
    model_params.add_argument('--word-min-count-target', required=False, type=int, default=1,
                              help='Minimum frequency of words to be included in target vocabulary. '
                                   'Default: %(default)s')


def add_vocab_args(required, optional):
    required.add_argument('-ts', '--train-source', required=True, type=str, help='Training source file.')
    required.add_argument('-tt', '--train-target', required=True, type=str, help='Training target file.')
    add_optional_args(optional)


def build_from_paths(input_source: str, input_target: str, single_vocab: bool = False, num_words_source: int = 50000,
                     num_words_target: int = 50000, min_count_source: int = 1, min_count_target: int = 1) -> (
        Dict[str, int], Dict[str, int]):
    """
    Creates vocabulary from paths to a file in sentence-per-line format. A sentence is just a whitespace delimited
    list of tokens. Note that special symbols like the beginning of sentence (BOS) symbol will be added to the
    vocabulary.
    :param input_target: Input original target file path.
    :param single_vocab: to build single vocabulary for source and target or not.
    :param num_words_source: number of max vocabulary on source side.
    :param num_words_target: number of max vocabulary on target side.
    :param min_count_source: Minimum frequency to include a word in source vocabulary.
    :param min_count_target: Minimum frequency to include a word in target vocabulary.
    :param input_source: Input original sour file path.
    :return: Word-to-id mapping.
    """
    with ExitStack() as stack:
        logger.info("Building vocabulary from dataset: %s and %s", input_source, input_target)
        files = (stack.enter_context(smart_open(path)) for path in [input_source, input_target])
        return build_vocab(*files, single_vocab=single_vocab, num_words_source=num_words_source,
                           num_words_target=num_words_target,
                           min_count_source=min_count_source,
                           min_count_target=min_count_target)


def read_worker(q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        int_source, int_target = deq
        record = list_to_record_bytes(int_source, int_target)
        q_out.put(record)


def write_worker(q_out, output_file):
    with open(output_file, "wb") as f:
        while True:
            deq = q_out.get()
            if deq is None:
                break
            write_recordio(f, deq)


def write_to_file(input_source: str, input_target: str, output_file: str, vocab_source: Dict[str, int],
                  vocab_target: Dict[str, int], file_type: str = "train"):
    """
    Converts the input strings to integers. Processes all the input files and writes into a single file each
    line if which is a list of integers.
    :param input_source: input original source file path (parallel corpus).
    :param input_target: input original target file path (parallel corpus).
    :param output_file: Path of output file to which the processed input file will be written
    :param vocab_source: String to Integer mapping of source vocabulary
    :param vocab_target: String to Integer mapping of target vocabulary
    """
    num_read_workers = max(multiprocessing.cpu_count() - 1, 1)
    logger.info('Spawning %s encoding worker(s) for encoding %s datasets!', str(num_read_workers), file_type)
    q_in = [multiprocessing.Queue() for i in range(num_read_workers)]
    q_out = multiprocessing.Queue()
    read_process = [multiprocessing.Process(target=read_worker,
                                            args=(q_in[i], q_out)) for i in range(num_read_workers)]
    for p in read_process:
        p.start()
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, output_file))
    write_process.start()

    lines_ignored = 0
    lines_processed = 0

    with ExitStack() as stack:
        files = (stack.enter_context(smart_open(path)) for path in [input_source, input_target])
        for line_source, line_target in zip(*files):
            if line_source.strip() == "" or line_target.strip() == "":
                lines_ignored += 1
                continue
            int_source = [vocab_source.get(token, vocab_source[UNK_SYMBOL]) for token in get_tokens(line_source)]
            int_target = [vocab_target.get(token, vocab_target[UNK_SYMBOL]) for token in get_tokens(line_target)]
            item = (int_source, int_target)
            q_in[lines_processed % len(q_in)].put(item)
            lines_processed += 1

    logger.info("""Processed %s lines for encoding to protobuf. %s lines were ignored as they didn't have
                any content in either the source or the target file!""", lines_processed, lines_ignored)

    logger.info('Completed writing the encoding queue!')
    for q in q_in:
        q.put(None)
    for p in read_process:
        p.join()

    logger.info('Encoding finished! Writing records to "%s"', output_file)
    q_out.put(None)
    write_process.join()

    logger.info('Processed input and saved to "%s"', output_file)


def prune_vocab(raw_vocab, num_words, min_count):
    # For words with the same count, they will be ordered reverse alphabetically.
    # Not an issue since we only care for consistency
    pruned_vocab = sorted(((c, w) for w, c in raw_vocab.items() if c >= min_count), reverse=True)
    # logger.info("Pruned vocabulary: %d types (min frequency %d)", len(pruned_vocab), min_count)

    vocab = islice((w for c, w in pruned_vocab), num_words)

    word_to_id = {word: idx for idx, word in enumerate(chain(VOCAB_SYMBOLS, vocab))}
    logger.info("Final vocabulary: %d types (min frequency %d, top %d types)", len(word_to_id), min_count, num_words)

    # Important: pad symbol becomes index 0
    assert word_to_id[PAD_SYMBOL] == PAD_ID
    return word_to_id


def build_vocab(data_source: Iterable[str], data_target: Iterable[str], single_vocab: bool = False,
                num_words_source: int = 50000, num_words_target: int = 50000, min_count_source: int = 1,
                min_count_target: int = 1) -> (Dict[str, int], Dict[str, int]):
    """
    Creates a vocabulary mapping from words to ids. Increasing integer ids are assigned by word frequency,
    using lexical sorting as a tie breaker. The only exception to this are special symbols such as the padding symbol
    (PAD).
    :param data_source: Sequence of source sentences containing whitespace delimited tokens.
    :param data_target: Sequence of target sentences containing whitespace delimited tokens.
    :param single_vocab: Whether to create a single vocab or not.
    :param num_words_source: Maximum number of words in the vocabulary for source side.
    :param num_words_target: Maximum number of words in the vocabulary for target side.
    :param min_count_source: Minimum frequency to include a word in source vocabulary.
    :param min_count_target: Minimum frequency to include a word in target vocabulary.
    :return: Word-to-id mapping.
    """
    vocab_symbols_set = set(VOCAB_SYMBOLS)

    if single_vocab:
        data = chain(data_source, data_target)
        raw_vocab = Counter(token for line in data for token in get_tokens(line) if token not in vocab_symbols_set)
        logger.info("Initial vocabulary: %d types" % len(raw_vocab))
        return prune_vocab(raw_vocab, num_words_source, min_count_source), None
    else:
        raw_vocab_source = Counter(token for line in data_source for token in get_tokens(line) if token not in
                                   vocab_symbols_set)
        raw_vocab_target = Counter(token for line in data_target for token in get_tokens(line) if token not in
                                   vocab_symbols_set)

        return (prune_vocab(raw_vocab_source, num_words_source, min_count_source),
                prune_vocab(raw_vocab_target, num_words_target, min_count_target))


def vocab_to_pickle(vocab: Mapping, path: str):
    """
    Saves vocabulary in pickle format.
    :param vocab: Vocabulary mapping.
    :param path: Output file path.
    """
    with open(path, 'wb') as out:
        pickle.dump(vocab, out)
        logger.info('Vocabulary saved to "%s"', path)


def vocab_to_json(vocab: Mapping, path: str):
    """
    Saves vocabulary in human-readable json.
    :param vocab: Vocabulary mapping.
    :param path: Output file path.
    """
    with open(path, "w") as out:
        json.dump(vocab, out, indent=4, ensure_ascii=False)
        logger.info('Vocabulary saved to "%s"', path)


def vocab_from_json_or_pickle(path) -> Dict:
    """
    Try loading the json version of the vocab and fall back to pickle for backwards compatibility.
    :param path: Path to vocab without the json suffix. If it exists the `path` + '.json' will be loaded as a JSON
        object and otherwise `path` is loaded as a pickle object.
    :return: The loaded vocabulary.
    """
    if os.path.exists(path + JSON_SUFFIX):
        return vocab_from_json(path + JSON_SUFFIX)
    else:
        return vocab_from_pickle(path)


def vocab_from_pickle(path: str) -> Dict:
    """
    Load vocab from pickle
    :param path: Path to pickle file containing the vocabulary.
    :return: The loaded vocabulary.
    """
    with open(path, 'rb') as inp:
        vocab = pickle.load(inp)
        logger.info('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab


def vocab_from_json(path: str) -> Dict:
    """
    Load vocab from JSON
    :param path: Path to json file containing the vocabulary.
    :return: The loaded vocabulary.
    """
    with open(path, encoding=VOCAB_ENCODING) as inp:
        vocab = json.load(inp)
        logger.info('Vocabulary (%d words) loaded from "%s"', len(vocab), path)
        return vocab


def reverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """
    Returns value-to-key mapping from key-to-value-mapping.
    :param vocab: Key to value mapping.
    :return: A mapping from values to keys.
    """
    return {v: k for k, v in vocab.items()}


def main():
    params = argparse.ArgumentParser(description='CLI to build vocabulary and pre-process input file.')
    required = params.add_argument_group('required arguments')
    add_vocab_args(required, params)
    args = params.parse_args()

    if not args.vocab_source_json or not args.vocab_target_json:
        vocab_source, vocab_target = build_from_paths(input_source=args.train_source, input_target=args.train_target,
                                                      single_vocab=args.single_vocab,
                                                      num_words_source=args.num_words_source,
                                                      num_words_target=args.num_words_target,
                                                      min_count_source=args.word_min_count_source,
                                                      min_count_target=args.word_min_count_target)
        logger.info("Source vocabulary size: %d ", len(vocab_source))
        vocab_to_json(vocab_source, "vocab.src" + JSON_SUFFIX)

        if not vocab_target:
            vocab_target = vocab_source
        logger.info("Target vocabulary size: %d ", len(vocab_target))
        vocab_to_json(vocab_target, "vocab.trg" + JSON_SUFFIX)
    else:
        vocab_source = vocab_from_json(args.vocab_source_json)
        vocab_target = vocab_from_json(args.vocab_target_json)

    vocab_target = vocab_target or vocab_source
    write_to_file(args.train_source, args.train_target, args.train_output, vocab_source, vocab_target)

    if args.val_source and args.val_target:
        write_to_file(args.val_source, args.val_target, args.val_output, vocab_source, vocab_target, "validation")


if __name__ == "__main__":
    main()
