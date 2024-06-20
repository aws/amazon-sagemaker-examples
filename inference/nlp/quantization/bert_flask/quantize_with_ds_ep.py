import argparse
import os
from typing import Tuple
import collections
import numpy as np
import torch
import intel_extension_for_pytorch as ipex
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    default_data_collator,
    BertTokenizer
)

__MODEL_DICT__ = dict()
__MODEL_FP32_DICT__ = dict()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Question Answering task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="squad",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences"
            " longer than this will be truncated, sequences shorter will be padded if"
            " `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed"
            " because the start and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="csarron/bert-base-uncased-squad-v1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the evaluation.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=(
            "When splitting up a long document into chunks how much stride to take"
            " between chunks."
        ),
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help=(
            "The total number of n-best predictions to generate when looking for an"
            " answer."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./",
        help="Where to store the quantized model.",
    )
    args = parser.parse_args()
    return args


def preprare_dataset(args):
    print("Prepraring dataset...")
    raw_datasets = load_dataset(args.dataset_name, None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    column_names = raw_datasets["train"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    tokenizer.save_pretrained(save_directory=args.model_path)

    def prepare_validation_features(examples):
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_examples = raw_datasets["validation"]
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )
    data_collator = default_data_collator
    eval_dataset_for_model = eval_dataset.remove_columns(
        ["example_id", "offset_mapping"]
    )
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size
    )
    return answer_column_name, eval_examples, eval_dataset, eval_dataloader

def TraceAndSave(args,model,eval_dataloader):
    model.eval()
    # Construct jit inputs based on example tensors
    jit_inputs = []
    example_batch = next(iter(eval_dataloader))
    for key in example_batch:
        example_tensor = torch.ones_like(example_batch[key])
        jit_inputs.append(example_tensor)
    jit_inputs = tuple(jit_inputs)
    
    with torch.no_grad():
        model = torch.jit.trace(model, jit_inputs, check_trace=False, strict=False)
        model = torch.jit.freeze(model)
    model.save(os.path.join(args.model_path, "model_fp32.pt"))

def IPEX_quantize(args, model, eval_dataloader):
    model.eval()
    conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_affine)
    #  Here we use dataset samples for calibations
    print("Doing calibration...")
    for step, batch in enumerate(eval_dataloader):
        print("Calibration step-", step)
        with torch.no_grad():
            # conf will be updated with observed statistics during calibrating with the dataset
            with ipex.quantization.calibrate(conf):
                model(**batch)
        if step == 5:
            break
    # [Optional] You can save this calibration file for later use
    # conf.save('int8_conf.json')

    # Construct jit inputs based on example tensors
    jit_inputs = []
    example_batch = next(iter(eval_dataloader))
    for key in example_batch:
        example_tensor = torch.ones_like(example_batch[key])
        jit_inputs.append(example_tensor)
    jit_inputs = tuple(jit_inputs)
    # Converting Quantization model:
    print("Doing model converting...")
    with torch.no_grad():
        model = ipex.quantization.convert(model, conf, jit_inputs)
    # Two iterations to enable fusions
    with torch.no_grad():
        model(**example_batch)
        model(**example_batch)
    # Save quantized model
    model.save(os.path.join(args.model_path, "model_int8.pt"))


def model_fn(args, eval_dataloader):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,torchscript=True,
        return_dict=False,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    # to save FP32 model
    TraceAndSave(args,model,eval_dataloader)
    # To generate the int8 quantized model and then save that next.
    IPEX_quantize(args, model, eval_dataloader)
    # model.save_pretrained(os.path.join(args.model_path, "model.pt"))
    model_path = os.path.join(args.model_path, "model_int8.pt")
    model = torch.jit.load(model_path)
    return model


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    all_predictions = collections.OrderedDict()
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_prediction = None
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


def model_fn_ep(model_dir):
    global __MODEL_DICT__
    if __MODEL_DICT__: 
        print("Model INT8 already loaded")
    else:
        print("Loading Model INT8")
        model_path = os.path.join(model_dir, 'model_int8.pt')    
        model = torch.jit.load(model_path) 
        model = model.to('cpu')
        tokenizer = BertTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1", use_fast=True)
        model_dict = {'model': model, 'tokenizer':tokenizer}
        __MODEL_DICT__ = model_dict
    return __MODEL_DICT__

def model_fn_ep_fp32(model_dir):
    global __MODEL_FP32_DICT__
    if __MODEL_FP32_DICT__: 
        print("Model FP32 already loaded")
    else:
        print("Loading Model FP32")
        model_path = os.path.join(model_dir, 'model_fp32.pt')    
        model_fp32 = torch.jit.load(model_path)
        model_fp32 = model_fp32.to('cpu')
        tokenizer = BertTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1", use_fast=True)
        model_fp32_dict = {'model': model_fp32, 'tokenizer':tokenizer}
        __MODEL_FP32_DICT__ = model_fp32_dict
    return __MODEL_FP32_DICT__

def predict_fn_ep(model_dict, input_data, context):       
    """
    Apply model to the incoming request
    """

    tokenizer = model_dict['tokenizer']
    model = model_dict['model']

    encoded_input = tokenizer.encode_plus(input_data, context, return_tensors='pt', max_length=128, padding='max_length', truncation=True)              
    output=model(**encoded_input) # This is specific to Inferentia
    answer_text = str(output[0])

    answer_start = torch.argmax(output[0])
    answer_end = torch.argmax(output[1])+1
    if (answer_end > answer_start):
        answer_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0][answer_start:answer_end]))
    else:
        answer_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0][answer_start:]))

    return answer_text;
  


def predict_fn(
    args, model, answer_column_name, eval_examples, eval_dataset, eval_dataloader
):
    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
        )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        for _, output_logit in enumerate(start_or_end_logits):
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]
            step += batch_size
        return logits_concat

    print("***** Running Evaluation *****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.batch_size}")
    all_start_logits = []
    all_end_logits = []
    for _, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs[0]
            end_logits = outputs[1]

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = create_and_fill_np_array(
        all_start_logits, eval_dataset, max_len
    )
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
    eval_metric = metric.compute(
        predictions=prediction.predictions, references=prediction.label_ids
    )
    print(f"Evaluation metrics: {eval_metric}")


def main():
    args = parse_args()
    print("***** Running fine-tuned Bert-base inference for Question Answering task with IPEX quantization *****")
    # Preparing the datasets
    answer_column_name, eval_examples, eval_dataset, eval_dataloader = preprare_dataset(
        args
    )

    # Preparing the ipex quantized model
    model = model_fn(args, eval_dataloader)
    # Doing model inference
    predict_fn(
        args, model, answer_column_name, eval_examples, eval_dataset, eval_dataloader
    )

    print("***** Test End Point***")
    model_dict = model_fn_ep(args.model_path)
    context= ("The Panthers finished the regular season with a 15-1 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP)."
    " They defeated the Arizona Cardinals 49-15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995."
    "The Broncos finished the regular season with a 12-4 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20-18 in the AFC Championship Game."
    " They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl.")
    question = "Who denied Patriots?"
    #question = "How many appearances have the Denver Broncos made in the Super Bowl?"
    answer_text = predict_fn_ep(model_dict, question, context)
    print("Question:", question, "Answer:", answer_text)
if __name__ == "__main__":
    main()
