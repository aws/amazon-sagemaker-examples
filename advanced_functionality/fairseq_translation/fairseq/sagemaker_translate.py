#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import copy
import json
import logging
import os
import sys
from collections import namedtuple

import numpy as np
import torch
from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator

Batch = namedtuple("Batch", "srcs tokens lengths")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


JSON_CONTENT_TYPE = "application/json"

logger = logging.getLogger(__name__)


def model_fn(model_dir):

    model_name = "checkpoint_best.pt"
    model_path = os.path.join(model_dir, model_name)

    logger.info("Loading the model")
    with open(model_path, "rb") as f:
        model_info = torch.load(f, map_location=torch.device("cpu"))

    # Will be overidden by the model_info['args'] - need to keep for pre-trained models
    parser = options.get_generation_parser(interactive=True)
    # get args for FairSeq by converting the hyperparameters as if they were command-line arguments
    argv_copy = copy.deepcopy(sys.argv)
    # remove the modifications we did in the command-line arguments
    sys.argv[1:] = ["--path", model_path, model_dir]
    args = options.parse_args_and_arch(parser)
    # restore previous command-line args
    sys.argv = argv_copy

    saved_args = model_info["args"]
    for key, value in vars(saved_args).items():
        setattr(args, key, value)

    args.data = [model_dir]
    print(args)

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current device: {}".format(device))

    model_paths = [os.path.join(model_dir, model_name)]
    models, model_args = utils.load_ensemble_for_inference(
        model_paths, task, model_arg_overrides={}
    )

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Initialize generator
    translator = SequenceGenerator(
        models,
        tgt_dict,
        beam_size=args.beam,
        minlen=args.min_len,
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen,
        unk_penalty=args.unkpen,
        sampling=args.sampling,
        sampling_topk=args.sampling_topk,
        sampling_temperature=args.sampling_temperature,
        diverse_beam_groups=args.diverse_beam_groups,
        diverse_beam_strength=args.diverse_beam_strength,
    )

    if device.type == "cuda":
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    # align_dict = utils.load_align_dict(args.replace_unk)
    align_dict = utils.load_align_dict(None)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    return dict(
        translator=translator,
        task=task,
        max_positions=max_positions,
        align_dict=align_dict,
        tgt_dict=tgt_dict,
        args=args,
        device=device,
    )


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info("Deserializing the input data.")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    raise Exception("Requested unsupported ContentType in content_type: " + content_type)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info("Serializing the generated output.")
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception("Requested unsupported ContentType in Accept: " + accept)


def predict_fn(input_data, model):
    args = model["args"]
    task = model["task"]
    max_positions = model["max_positions"]
    device = model["device"]
    translator = model["translator"]
    align_dict = model["align_dict"]
    tgt_dict = model["tgt_dict"]

    inputs = [input_data]

    indices = []
    results = []
    for batch, batch_indices in make_batches(inputs, args, task, max_positions):
        indices.extend(batch_indices)
        results += process_batch(batch, translator, device, args, align_dict, tgt_dict)

    r = []
    for i in np.argsort(indices):
        result = results[i]
        # print(result.src_str)
        for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
            r.append(hypo)
            # print(hypo)
            # print(pos_scores)
            if align is not None:
                print(align)
    return "\n".join(r)


##################################
# Helper functions
##################################


def process_batch(batch, translator, device, args, align_dict, tgt_dict):
    tokens = batch.tokens.to(device)
    lengths = batch.lengths.to(device)

    encoder_input = {"src_tokens": tokens, "src_lengths": lengths}
    translations = translator.generate(
        encoder_input,
        maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
    )

    return [
        make_result(batch.srcs[i], t, align_dict, tgt_dict, args)
        for i, t in enumerate(translations)
    ]


def make_result(src_str, hypos, align_dict, tgt_dict, args):
    result = Translation(
        src_str="O\t{}".format(src_str),
        hypos=[],
        pos_scores=[],
        alignments=[],
    )

    # Process top predictions
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo["tokens"].int().cpu(),
            src_str=src_str,
            alignment=hypo["alignment"].int().cpu() if hypo["alignment"] is not None else None,
            align_dict=align_dict,
            tgt_dict=tgt_dict,
            remove_bpe=args.remove_bpe,
        )
        # result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
        # only get the traduction, not the score
        result.hypos.append(hypo_str)
        result.pos_scores.append(
            "P\t{}".format(
                " ".join(
                    map(
                        lambda x: "{:.4f}".format(x),
                        hypo["positional_scores"].tolist(),
                    )
                )
            )
        )
        result.alignments.append(
            "A\t{}".format(" ".join(map(lambda x: str(utils.item(x)), alignment)))
            if args.print_alignment
            else None
        )
    return result


def make_batches(lines, args, task, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(tokens, lengths, task.source_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch["id"]],
            tokens=batch["net_input"]["src_tokens"],
            lengths=batch["net_input"]["src_lengths"],
        ), batch["id"]
