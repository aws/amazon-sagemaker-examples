import collections
from types import SimpleNamespace
RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
import math

def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def preprocess_tokenized_text(doc_tokens, query_tokens, tokenizer, 
                              max_seq_length, max_query_length):
    """ converts an example into a feature """
    
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    
    # truncate if too long
    length = len(all_doc_tokens)
    length = min(length, max_tokens_for_doc)
    
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    for i in range(length):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        token_is_max_context[len(tokens)] = True
        tokens.append(all_doc_tokens[i])
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    tensors_for_inference = {
                             'input_ids': input_ids, 
                             'input_mask': input_mask, 
                             'segment_ids': segment_ids
                            }
    tensors_for_inference = SimpleNamespace(**tensors_for_inference)
    
    tokens_for_postprocessing = {
                                 'tokens': tokens,
                                 'token_to_orig_map': token_to_orig_map,
                                 'token_is_max_context': token_is_max_context
                                }
    tokens_for_postprocessing = SimpleNamespace(**tokens_for_postprocessing)
    
    return tensors_for_inference, tokens_for_postprocessing

def get_predictions(doc_tokens, tokens_for_postprocessing, 
                    start_logits, end_logits, n_best_size, 
                    max_answer_length, do_lower_case, 
                    can_give_negative_answer, null_score_diff_threshold):
    """ Write final predictions to the json file and log-odds of null if needed. """
    result = RawResult(start_logits=start_logits, end_logits=end_logits)
    
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", 
        ["start_index", "end_index", "start_logit", "end_logit"])
    
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    
    start_indices = _get_indices_of_largest_logits(result.start_logits)
    end_indices = _get_indices_of_largest_logits(result.end_logits)
    # if we could have irrelevant answers, get the min score of irrelevant
    if can_give_negative_answer:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
            score_null = feature_null_score
            null_start_logit = result.start_logits[0]
            null_end_logit = result.end_logits[0]
    for start_index in start_indices:
        for end_index in end_indices:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(tokens_for_postprocessing.tokens):
                continue
            if end_index >= len(tokens_for_postprocessing.tokens):
                continue
            if start_index not in tokens_for_postprocessing.token_to_orig_map:
                continue
            if end_index not in tokens_for_postprocessing.token_to_orig_map:
                continue
            if not tokens_for_postprocessing.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]
                    )
            )
    if can_give_negative_answer:
        prelim_predictions.append(
            _PrelimPrediction(
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit
            )
        )
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True
    )
    
    _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
    
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = tokens_for_postprocessing.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = tokens_for_postprocessing.token_to_orig_map[pred.start_index]
            orig_doc_end = tokens_for_postprocessing.token_to_orig_map[pred.end_index]
            orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)
            
            # de-tokenize WordPieces that have been split off. 
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")
            
            # clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)
            
            # get final text
            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue
            
            # mark it
            seen_predictions[final_text] = True
            
        else: # this is a null prediction
            final_text = ""
            seen_predictions[final_text] = True
        
        nbest.append(
            _NbestPrediction(
                text=final_text, 
                start_logit=pred.start_logit, 
                end_logit=pred.end_logit
            )
        )
    # if we didn't include the empty option in the n-best, include it 
    if can_give_negative_answer:
        if "" not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text="", 
                    start_logit=null_start_logit, 
                    end_logit=null_end_logit
                )
            )
        # In very rare edge cases we could only have single null prediction. 
        # So we just create a nonce prediction in this case to avoid failure. 
        if len(nbest) == 1:
            nbest.insert(0, _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))
    
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure. 
    if not nbest:
        nbest.append(_NbestPrediction(text="", start_logit=0.0, end_logit=0.0))
    
    assert len(nbest) >= 1
    
    # scoring
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry
    
    # get probabilities
    probs = _compute_softmax(total_scores)
    
    # nbest predictions into json format
    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)
    
    assert len(nbest_json) >= 1
    
    if can_give_negative_answer:
        # predict "unknown" iff ((score_null - score_of_best_non-null_entry) > threshold)
        score = best_non_null_entry.start_logit + best_non_null_entry.end_logit
        score_diff = score_null - score
        if score_diff > null_score_diff_threshold:
            nbest_json[0]['text'] = "unknown"
            # best_non_null_entry.text = "unknown"
    # 
    return nbest_json

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _get_indices_of_largest_logits(logits):
    """ sort logits and return the indices of the sorted array """
    indices_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    indices = map(lambda x: x[0], indices_and_score)
    indices = list(indices)
    return indices


def preprocess_text_input(context='Danielle is a girl who really loves her cat, Steve.', 
                         question='What cat does Danielle love?',
                         vocab_file='DeepLearningExamples/PyTorch/LanguageModeling/BERT/vocab/vocab',
                         max_seq_length=384, max_query_length=64, n_best_size=1, max_answer_length=30, 
                         null_score_diff_threshold=-11.0):
    tokenizer = BertTokenizer(vocab_file, do_lower_case=True, max_len=512)
    doc_tokens = context.split()
    query_tokens = tokenizer.tokenize(question)
    feature = preprocess_tokenized_text(doc_tokens, 
                                        query_tokens, 
                                        tokenizer, 
                                        max_seq_length=max_seq_length, 
                                        max_query_length=max_query_length)

    tensors_for_inference, tokens_for_postprocessing = feature

    input_ids = torch.tensor(tensors_for_inference.input_ids, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(tensors_for_inference.segment_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(tensors_for_inference.input_mask, dtype=torch.long).unsqueeze(0)
    return(input_ids, segments_ids, input_mask)