# -*- coding: utf-8 -*-
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
import argparse
import collections
import json
import logging
import math
import os
import pickle
import random
import time
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm
from transformers import BertConfig, BertTokenizer, AlbertTokenizer, AlbertConfig
from transformers.tokenization_bert import whitespace_tokenize
import re

tqdm.monitor_interval = 0

logger = logging.getLogger()

NQExample = collections.namedtuple("NQExample", [
    "qas_id", "question_text", "doc_tokens", "orig_answer_text",
    "start_position", "end_position", "long_position",
    "short_is_impossible", "long_is_impossible", "crop_start"])

Crop = collections.namedtuple("Crop", ["unique_id", "example_index", "doc_span_index",
                                       "tokens", "token_to_orig_map", "token_is_max_context",
                                       "input_ids", "attention_mask", "token_type_ids",
                                       # "p_mask",
                                       "paragraph_len", "start_position", "end_position",
                                       "long_position", "short_is_impossible", "long_is_impossible"])

LongAnswerCandidate = collections.namedtuple('LongAnswerCandidate', [
    'start_token', 'end_token', 'top_level'])

DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                          ["crop_index", "start_index", "end_index", "start_logit", "end_logit"])

NbestPrediction = collections.namedtuple("NbestPrediction", [
    "text", "start_logit", "end_logit",
    "start_index", "end_index",
    "orig_doc_start", "orig_doc_end", "crop_index"])

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits",
                                                 "long_logits"])

UNMAPPED = -123
CLS_INDEX = 0


def get_add_tokens(do_enumerate):
    """
       This function crea array con tutti i tag più gli speciali
       @param do_enumerate se è TRUE sono aggiunti anche i flag speciali
       @return array di tokens
       """

    tags = ['Dd', 'Dl', 'Dt', 'H1', 'H2', 'H3', 'Li', 'Ol', 'P', 'Table', 'Td', 'Th', 'Tr', 'Ul']
    opening_tags = [f'<{tag}>' for tag in tags]
    closing_tags = [f'</{tag}>' for tag in tags]
    added_tags = opening_tags + closing_tags
    # See `nq_to_sqaud.py` for special-tokens
    special_tokens = ['<P>', '<Table>']
    if do_enumerate:
        for special_token in special_tokens:
            for j in range(11):
                added_tags.append(f'<{special_token[1: -1]}{j}>')

    add_tokens = ['Td_colspan', 'Th_colspan', '``', '\'\'', '--']
    add_tokens = add_tokens + added_tags
    return add_tokens


def read_candidates(candidate_files, do_cache=True):
    assert isinstance(candidate_files, (tuple, list)), candidate_files
    for fn in candidate_files:
        assert os.path.exists(fn), f'Missing file {fn}'
    cache_fn = '../cache/candidates.pkl'

    candidates = {}  # Creo il dizionario dei candidati (che sia dalla cache o meno)
    question = {}
    if not os.path.exists(cache_fn):
        for fn in candidate_files:
            with open(fn) as f:
                for line in tqdm(f):  # tqdm è la progress bar
                    entry = json.loads(line)
                    example_id = str(entry['example_id'])
                    cnds = entry.pop('long_answer_candidates')
                    cnds = [LongAnswerCandidate(c['start_token'], c['end_token'],
                                                c['top_level']) for c in
                            cnds]  # le informazioni sono comprese nella long answer. Top_level mi dice se la
                    # risposta è dentro un'altra o meno
                    candidates[example_id] = cnds
                    question[example_id] = str(entry['question_text'])

        if do_cache:
            print("Saving candidates to pkl")
            with open(cache_fn, 'wb') as f:
                pickle.dump(candidates, f)
    else:
        print(f'Loading from cache: {cache_fn}')
        with open(cache_fn, 'rb') as f:
            candidates = pickle.load(f)

    return candidates, question


def is_whitespace(c):  # ok
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_nq_examples(input_file_or_data,
                     is_training):  # modo diverso per fare questo https://github.com/google/retrieval-qa-eval/blob
    # /master/nq_to_squad.py (?)

    """
    Read a NQ json file into a list of NQExample. Refer to `nq_to_squad.py`
    to convert the `simplified-nq-t*.jsonl` files to NQ json.
       This function returns NQExample
       @param input_file_or_data entries from jsonfile
       @param is_training boolean to Train or Evaluate
       @return a collection of NQ Examples
       """

    if isinstance(input_file_or_data, str):
        print("è File")
        with open(input_file_or_data, "r", encoding='utf-8') as f:
            input_data = json.load(f)["data"]

    else:
        input_data = input_file_or_data
    print("Dimensione: " + str(len(input_data)))
    for entry_index, entry in enumerate(tqdm(input_data, total=len(input_data))):
        # if entry_index >= 2:
        #     break
        assert len(entry["paragraphs"]) == 1
        paragraph = entry["paragraphs"][0]
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        assert len(paragraph["qas"]) == 1
        qa = paragraph["qas"][0]
        start_position = None
        end_position = None
        long_position = None
        orig_answer_text = None
        short_is_impossible = False
        long_is_impossible = False
        if is_training:
            short_is_impossible = qa["short_is_impossible"]
            short_answers = qa["short_answers"]
            if len(short_answers) >= 2:
                # logger.info(f"Choosing leftmost of "
                #     f"{len(short_answers)} short answer")
                short_answers = sorted(short_answers, key=lambda sa: sa["answer_start"])
                short_answers = short_answers[0: 1]

            if not short_is_impossible:
                answer = short_answers[0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[
                    answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly
                # recovered from the document. If this CAN'T
                # happen it's likely due to weird Unicode stuff
                # so we will just skip the example.
                #
                # Note that this means for training mode, every
                # example is NOT guaranteed to be preserved.
                actual_text = " ".join(doc_tokens[start_position:
                                                  end_position + 1])
                cleaned_answer_text = " ".join(
                    whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning(
                        "Could not find answer: '%s' vs. '%s'",
                        actual_text, cleaned_answer_text)
                    continue
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""

            long_is_impossible = qa["long_is_impossible"]
            long_answers = qa["long_answers"]
            if (len(long_answers) != 1) and not long_is_impossible:
                raise ValueError(f"For training, each question"
                                 f" should have exactly 1 long answer.")

            if not long_is_impossible:
                long_answer = long_answers[0]
                long_answer_offset = long_answer["answer_start"]
                long_position = char_to_word_offset[long_answer_offset]
            else:
                long_position = -1

            if not short_is_impossible and not long_is_impossible:
                assert long_position <= start_position

            if not short_is_impossible and long_is_impossible:
                assert False, f'Invalid pair short, long pair'

        example = NQExample(
            qas_id=qa["id"],
            question_text=qa["question"],
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            long_position=long_position,
            short_is_impossible=short_is_impossible,
            long_is_impossible=long_is_impossible,
            crop_start=qa["crop_start"])

        yield example


def get_spans(doc_stride, max_tokens_for_doc, max_len):  # non chiaro, ma lo usa sotto
    doc_spans = []
    start_offset = 0
    while start_offset < max_len:
        length = max_len - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(DocSpan(start=start_offset, length=length))
        if start_offset + length == max_len:
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def convert_examples_to_crops(examples_gen, tokenizer, max_seq_length,
                              doc_stride, max_query_length, is_training,
                              cls_token='[CLS]', sep_token='[SEP]', pad_id=0,
                              sequence_a_segment_id=0,
                              sequence_b_segment_id=1,
                              cls_token_segment_id=0,
                              pad_token_segment_id=0,
                              mask_padding_with_zero=True,
                              p_keep_impossible=None,
                              sep_token_extra=False):
    """
           Trasforma gli NQExample in Crops
           @param examples_gen gli example da convertire in crops
           @param tokenizer
           @param max_seq_length lunghezza massima (FORSE) fra domanda+risposta
           @:var
           @return list of Crops
           """
    assert p_keep_impossible is not None, '`p_keep_impossible` is required'
    unique_id = 1000000000
    num_short_pos, num_short_neg = 0, 0
    num_long_pos, num_long_neg = 0, 0
    sub_token_cache = {}

    crops = []
    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            print('Converting %s: short_pos %s short_neg %s'
                  ' long_pos %s long_neg %s',
                  example_index, num_short_pos, num_short_neg,
                  num_long_pos, num_long_neg)

        query_tokens = tokenizer.tokenize(example.question_text)  # tokenizzo la domanda e se troppo lunga la tronco
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # this takes the longest!
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        for i, token in enumerate(example.doc_tokens):
            # take the work token
            orig_to_tok_index.append(len(all_doc_tokens))
            # map it to the number
            sub_tokens = sub_token_cache.get(token)
            if sub_tokens is None:  # tokenizzo i token di ogni documento(?)
                sub_tokens = tokenizer.tokenize(token)
                sub_token_cache[token] = sub_tokens
            # Qui succede questo:
            # un token tipo 'Klementieff' diventa 4 subtoken : ['▁kle', 'ment', 'i', 'eff']
            # quindi mappi tutti i subtoken corrispondenti al token del testo originale corrispondente a Klementieff
            tok_to_orig_index.extend([i for _ in range(len(sub_tokens))])
            all_doc_tokens.extend(sub_tokens)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.short_is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        # assegno le posizioni se è possibile rispondere alla domanda corta e sono in TRAIING
        if is_training and not example.short_is_impossible:
            # qua traduco da orignal a token in quanto posso essere shifato dal tokenizer
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[  # why
                                       example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

        tok_long_position = None
        if is_training and example.long_is_impossible:
            tok_long_position = -1

        if is_training and not example.long_is_impossible:
            tok_long_position = orig_to_tok_index[example.long_position]

        # For Bert: [CLS] question [SEP] paragraph [SEP]
        special_tokens_count = 3
        if sep_token_extra:
            # For Roberta: <s> question </s> </s> paragraph </s>
            special_tokens_count += 1
        max_tokens_for_doc = max_seq_length - len(query_tokens) - special_tokens_count
        assert max_tokens_for_doc > 0
        # We can have documents that are longer than the maximum
        # sequence length. To deal with this we do a sliding window
        # approach, where we take chunks of the up to our max length
        # with a stride of `doc_stride`.
        doc_spans = get_spans(doc_stride, max_tokens_for_doc, len(all_doc_tokens))
        for doc_span_index, doc_span in enumerate(doc_spans):
            # Tokens are constructed as: CLS Query SEP Paragraph SEP
            tokens = []
            token_to_orig_map = UNMAPPED * np.ones((max_seq_length,), dtype=np.int32)
            token_is_max_context = np.zeros((max_seq_length,), dtype=np.bool)
            token_type_ids = []

            # p_mask: mask with 1 for token than cannot be in the
            # answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token
            # (set to 0) (not sure why...)
            # p_mask = []

            short_is_impossible = example.short_is_impossible
            start_position = None
            end_position = None
            special_tokens_offset = special_tokens_count - 1
            doc_offset = len(query_tokens) + special_tokens_offset
            if is_training and not short_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    start_position = 0
                    end_position = 0
                    short_is_impossible = True
                else:
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            long_is_impossible = example.long_is_impossible
            long_position = None
            if is_training and not long_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                # out of span
                if not (tok_long_position >= doc_start and tok_long_position <= doc_end):
                    long_position = 0
                    long_is_impossible = True
                else:
                    long_position = tok_long_position - doc_start + doc_offset

            # drop impossible samples -> see Alberti
            if long_is_impossible:
                if np.random.rand() > p_keep_impossible:
                    continue

            # CLS token at the beginning
            tokens.append(cls_token)
            token_type_ids.append(cls_token_segment_id)
            # p_mask.append(0)  # can be answer

            # Query
            tokens += query_tokens
            token_type_ids += [sequence_a_segment_id] * len(query_tokens)
            # p_mask += [1] * len(query_tokens)  # can not be answer

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_a_segment_id)
            # p_mask.append(1)  # can not be answer
            if sep_token_extra:
                tokens.append(sep_token)
                token_type_ids.append(sequence_a_segment_id)
                # p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                # We add `example.crop_start` as the original document
                # is already shifted
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                                                     split_token_index] + example.crop_start

                token_is_max_context[len(tokens)] = check_is_max_context(doc_spans,
                                                                         doc_span_index, split_token_index)
                tokens.append(all_doc_tokens[split_token_index])
                token_type_ids.append(sequence_b_segment_id)
                # p_mask.append(0)  # can be answer

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_b_segment_id)
            # p_mask.append(1)  # can not be answer

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_id)
                attention_mask.append(0 if mask_padding_with_zero else 1)
                token_type_ids.append(pad_token_segment_id)

            # reduce memory, only input_ids needs more bits
            input_ids = np.array(input_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.bool)
            token_type_ids = np.array(token_type_ids, dtype=np.uint8)

            if is_training and short_is_impossible:
                start_position = CLS_INDEX
                end_position = CLS_INDEX

            if is_training and long_is_impossible:
                long_position = CLS_INDEX

            if short_is_impossible:
                num_short_neg += 1
            else:
                num_short_pos += 1

            if long_is_impossible:
                num_long_neg += 1
            else:
                num_long_pos += 1

            crop = Crop(
                unique_id=unique_id,
                example_index=example_index,
                # example_index=example.qas_id,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                # p_mask=p_mask,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position,
                long_position=long_position,
                short_is_impossible=short_is_impossible,
                long_is_impossible=long_is_impossible)
            crops.append(crop)
            unique_id += 1

    return crops


def check_is_max_context(doc_spans, cur_span_index, position):
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
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def clean_text(tok_text):
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")
    tok_text = tok_text.replace("<p>", " ")
    tok_text = tok_text.replace("</p>", " ")
    tok_text = re.sub('\[CLS].*?\[SEP]', '', tok_text, flags=re.DOTALL)
    tok_text = tok_text.replace("[SEP]", " ")
    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def get_nbest(prelim_predictions, crops, n_best_size):
    """
    This function extracts the text for the n_best_size best long 
    answers from the sorted predictions in prelim_predictions

    @param prelim_predictions list of crop indexes containing the
        start position of each long answer produced
    @param crops list of crops
    @param example index of the current example
    @param n_best_size number of top predictions to be returned
    """
    meno = 0

    seen, nbest = set(), []
    # for each prediction (they are already sorted according to confidence)
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        # take the crop that contains the start position

        crop = crops[pred.crop_index]
        # initialize start and end position
        orig_doc_start, orig_doc_end = -1, -1
        # if the start position is -1 it means that the question is considered
        # as unanswerable
        if pred.start_index > 0:
            # print(f" start:{crop.tokens[pred.start_index]}")
            # if this prediction is a short answer
            if pred.end_index != -1:
                start_index = pred.start_index
                end_index = pred.end_index
                start_crop_index = pred.crop_index
                end_crop_index = pred.crop_index

                tokens = crop.tokens[pred.start_index: pred.end_index + 1]

            # if it is a long answer find best start and end position
            else:
                # find the correct start position (possibly in previous crops)
                tags = ['dd', 'dl', 'dt', 'h1', 'h2', 'h3', 'li', 'ol', 'p', 'table', 'td', 'th', 'tr', 'ul']
                tags_dict = {f'<{tag}>': f'</{tag}>' for tag in tags}
                tags_reverse_dict = {f'</{tag}>': f'<{tag}>' for tag in tags}
                start_index = pred.start_index
                start_crop_index = pred.crop_index
                # if the start index of the prediction does not start
                # with an opening tag
                if crop.tokens[start_index] not in tags_dict.keys():
                    # find backwards the first tag that opens and does not close
                    while start_crop_index >= 0:
                        # take all the LAST occurrences of each opening tag
                        # in the list of all admittable opening tags
                        last_opening_positions = []
                        crop_tokens = np.array(crops[start_crop_index].tokens)
                        for closing_tag in tags_dict.values():
                            if start_crop_index == pred.crop_index:
                                closing_tag_positions = np.where(crop_tokens[:pred.start_index] == closing_tag)
                            else:
                                closing_tag_positions = np.where(crop_tokens == closing_tag)
                            # if there is not a closing tag satisfying the constraints
                            # go to next opening tag in dictionary
                            if closing_tag_positions[0].size <= 0:
                                continue

                            last_closing_tag_index = max(closing_tag_positions[0])
                            last_opening_tag_index = -1
                            opening_tag = tags_reverse_dict[closing_tag]

                            # check if there is an opening tag (corresponding to closing_tag)
                            # at an index bigger than last_closing_tag_index but also smaller
                            # than pred.start_index
                            if start_crop_index == pred.crop_index:
                                tmp_array = np.where(
                                    crop_tokens[last_closing_tag_index:pred.start_index] == opening_tag)
                            else:
                                tmp_array = np.where(crop_tokens[last_closing_tag_index:] == opening_tag)
                            opening_tag_positions = [i + last_closing_tag_index for i in tmp_array[0]]
                            # if there is not a opening tag satisfying the constraints
                            # go to next opening tag in dictionary
                            if len(opening_tag_positions) <= 0:
                                continue

                            # take the maximum occurrence of such tag
                            last_opening_tag_index = max(opening_tag_positions)
                            last_opening_positions.append(last_opening_tag_index)

                        # if there is at least an opening tag that is not followed
                        # by a closing tag in this crop take the last one and stop
                        # the backwards scanning of the crops list
                        if len(last_opening_positions) > 0:
                            start_index = max(last_opening_positions)
                            break
                        start_crop_index = start_crop_index - 1

                    # if start_index is the predicted one but start_crop_index < 0
                    # it means that no admissible opening tag was found
                    if start_index == pred.start_index and start_crop_index < 0:
                        # if start_index is not the last token of the crop take
                        # the subsequent token
                        start_crop_index = pred.crop_index
                        if start_index < len(crops[start_crop_index].tokens):
                            start_index = start_index + 1

                # at this point, we want to find the first occurrence of the closing
                # tag that corresponds to the opening tag AFTER the predicted start token
                # OBS: it is possible that no opening tag satisfying the requirements
                # was found. In that case, the closing tag cannot be found

                end_index = pred.start_index
                end_crop_index = pred.crop_index
                # if no opening tag was found, cut the answer ten tokens after the start
                if crops[start_crop_index].tokens[start_index] not in tags_dict.keys():
                    end_index = start_index + 10
                    if end_index > len(crops[start_crop_index].tokens):
                        end_index = len(crops[start_crop_index].tokens)

                else:
                    closing_tag = tags_dict[crops[start_crop_index].tokens[start_index]]

                    # take the first occurrence of the target closing tag
                    iter = 0
                    while end_crop_index < len(crops):
                        # if we are in the same crop of the prediction we need to be sure that
                        # the prediction is contained in the answer
                        if end_crop_index == start_crop_index:
                            tmp_array = np.where(np.array(crops[end_crop_index].tokens[start_index:]) == closing_tag)
                            # print(tag_position[0])
                            # tag_positions = (np.array([i+start_index for i in tag_position[0]]))
                            tag_positions = [i + start_index for i in tmp_array[0]]
                        else:
                            tmp_array = np.where(np.array(crops[end_crop_index].tokens) == closing_tag)
                            tag_positions = [i for i in tmp_array[0]]
                        if tag_positions:
                            end_index = min(tag_positions) + 1
                            break
                        end_crop_index = end_crop_index + 1
                        iter += 1

                    # if the last crop was reached
                    if end_crop_index == len(crops):
                        print("ultimo")
                        # if no closing tag was found
                        if end_index == pred.start_index:
                            # take the first 10 tokens after the start
                            end_index = start_index + 10
                            if end_index > len(crops[start_crop_index].tokens):
                                end_index = len(crops[start_crop_index].tokens)
                            end_crop_index = start_crop_index
                        else:
                            end_crop_index -= 1

                    # the couple (end_crop_index, end_index) uniquely identifies the
                    # predicted end position of both long and short answer

                    # if the closing tag was not found in the whole text take 10 tokens
                    if start_crop_index == end_crop_index and start_index == end_index:
                        end_index = start_index + 10
                        if end_index > len(crops[start_crop_index].tokens):
                            end_index = len(crops[start_crop_index].tokens)

                # take the tokens
                if start_crop_index == end_crop_index:
                    tokens = crops[start_crop_index].tokens[start_index:end_index]
                else:
                    # take all the tokens in the first crop from start position
                    tokens = crops[start_crop_index].tokens[start_index:]
                    # append all the tokens in subsequent crops
                    for crop_idx in range(start_crop_index + 1, end_crop_index):
                        tokens.extend(crops[crop_idx].tokens)
                    # we are at last crop
                    tokens.extend(crops[end_crop_index].tokens[:end_index])

            # get the text
            text = " ".join(tokens)
            text = clean_text(text)
            orig_doc_start = int(crops[start_crop_index].token_to_orig_map[start_index])
            orig_doc_end = int(crops[end_crop_index].token_to_orig_map[end_index])


        # if is unanswerable no text is selected
        else:
            print("lese no answerable")
            text = ""
            # Dummy assignment
            start_index = -1
            end_index = -1
            start_crop_index = -1
            end_crop_index = -1
            meno += 1
        # if the selected text is already present in other predictions continue
        if text in seen:
            continue
            # else add it to seen answers and append it to nbest
        seen.add(text)

        ###TODO: gli start ed end logit non li ho modificati, quindi non so se è
        ### corretto ritornarli così
        ### inoltre il parametro "crop_index" è mal definito, in quanto noi abbiamo
        ### l'indice del primo crop e dell'ultimo crop che potrebbero non coincidere
        nbest.append(NbestPrediction(
            text=text,
            start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=start_index, end_index=end_index,
            orig_doc_start=orig_doc_start, orig_doc_end=orig_doc_end,
            crop_index=start_crop_index))

    # Degenerate case. I never saw this happen. OH SHIIIEEET
    if len(nbest) in (0, 1):
        nbest.insert(0, NbestPrediction(text="empty",
                                        start_logit=0.0, end_logit=0.0,
                                        start_index=-1, end_index=-1,
                                        orig_doc_start=-1, orig_doc_end=-1,
                                        crop_index=UNMAPPED))

    assert len(nbest) >= 1
    # print(f"{meno} di {tipo_dom}")
    return nbest


def get_nbest_old(prelim_predictions, crops, n_best_size):
    seen, nbest = set(), []
    meno = 0
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        crop = crops[pred.crop_index]
        orig_doc_start, orig_doc_end = -1, -1
        if pred.start_index > 0:
            # Long answer has no end_index. We still generate some text to check
            if pred.end_index == -1:
                array_tmp = np.array(crop.tokens)
                indx_array = np.where(array_tmp == '</p>')
                try:
                    end_indx = indx_array[0][indx_array[0] > pred.start_index][0] + 1  # Prendo il primo indice più
                    # grande di start
                except:
                    end_indx = pred.start_index + 11
                tok_tokens = crop.tokens[
                             pred.start_index: end_indx]
            else:
                tok_tokens = crop.tokens[pred.start_index: pred.end_index + 1]

            tok_text = " ".join(tok_tokens)
            tok_text = clean_text(tok_text)

            orig_doc_start = int(crop.token_to_orig_map[pred.start_index])
            if pred.end_index == -1:
                orig_doc_end = orig_doc_start + 10
            else:
                orig_doc_end = int(crop.token_to_orig_map[pred.end_index])

            final_text = tok_text
            if final_text in seen:
                continue

        else:
            final_text = ""
            meno += 1
        seen.add(final_text)
        nbest.append(NbestPrediction(
            text=final_text,
            start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=pred.start_index, end_index=pred.end_index,
            orig_doc_start=orig_doc_start, orig_doc_end=orig_doc_end,
            crop_index=pred.crop_index))

    # Degenerate case. I never saw this happen.
    if len(nbest) in (0, 1):
        nbest.insert(0, NbestPrediction(text="empty",
                                        start_logit=0.0, end_logit=0.0,
                                        start_index=-1, end_index=-1,
                                        orig_doc_start=-1, orig_doc_end=-1,
                                        crop_index=UNMAPPED))
    # print(f"{meno} di ")
    assert len(nbest) >= 1
    return nbest


def write_predictions(examples_gen, all_crops, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      short_null_score_diff, long_null_score_diff, method):
    """
    Write final predictions to the json file and log-odds of null if needed.
    method:
        1) '' = default
        2) 'restoring' = if rejected and short IN long text ->taken
        3) 'matching' = taking the best short IN long token
    """
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    # create indexes
    example_index_to_crops = collections.defaultdict(list)
    for crop in all_crops:
        example_index_to_crops[crop.example_index].append(crop)
    unique_id_to_result = {result.unique_id: result for result in all_results}
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    short_num_empty, long_num_empty = 0, 0
    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            logger.info(f'[{example_index}]: {short_num_empty} short and {long_num_empty} long empty')
        crops = example_index_to_crops[example_index]

        short_prelim_predictions, long_prelim_predictions = [], []
        for crop_index, crop in enumerate(crops):
            assert crop.unique_id in unique_id_to_result, f"{crop.unique_id}"

            result = unique_id_to_result[crop.unique_id]
            # get the `n_best_size` largest indexes
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array#23734295
            start_indexes = np.argpartition(result.start_logits, -n_best_size)[-n_best_size:]
            start_indexes = [int(x) for x in start_indexes]
            end_indexes = np.argpartition(result.end_logits, -n_best_size)[-n_best_size:]
            end_indexes = [int(x) for x in end_indexes]

            # create short answers
            for start_index in start_indexes:
                if start_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[start_index] == UNMAPPED:
                    continue
                if not crop.token_is_max_context[start_index]:
                    continue

                for end_index in end_indexes:
                    if end_index >= len(crop.tokens):
                        continue
                    if crop.token_to_orig_map[end_index] == UNMAPPED:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    short_prelim_predictions.append(PrelimPrediction(
                        crop_index=crop_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

            long_indexes = np.argpartition(result.long_logits, -n_best_size)[-n_best_size:].tolist()
            for long_index in long_indexes:
                if long_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[long_index] == UNMAPPED:
                    continue
                # TODO(see--): Is this needed?
                # -> Yep helps both short and long by about 0.1
                if not crop.token_is_max_context[long_index]:
                    continue
                long_prelim_predictions.append(PrelimPrediction(
                    crop_index=crop_index,
                    start_index=long_index, end_index=-1,
                    start_logit=result.long_logits[long_index],
                    end_logit=result.long_logits[long_index]))

        short_prelim_predictions = sorted(short_prelim_predictions,
                                          key=lambda x: x.start_logit + x.end_logit, reverse=True)

        short_nbest = get_nbest_old(short_prelim_predictions, crops, n_best_size)

        long_prelim_predictions = sorted(long_prelim_predictions,
                                         key=lambda x: x.start_logit, reverse=True)

        long_nbest = get_nbest(long_prelim_predictions, crops, n_best_size)

        long_best_non_null = None
        for entry in long_nbest:
            if long_best_non_null is None:
                if entry.text != "":
                    long_best_non_null = entry
        short_best_non_null = None
        for entry in short_nbest:
            if short_best_non_null is None:
                if entry.text != "":
                    if method == 'matching' or method == 'mixed':
                        if entry.orig_doc_start >= long_best_non_null.orig_doc_start:
                            short_best_non_null = entry
                    else:
                        short_best_non_null = entry

        nbest_json = {'short': [], 'long': []}
        for kk, entries in [('short', short_nbest), ('long', long_nbest)]:
            for i, entry in enumerate(entries):
                output = {}
                output["text"] = entry.text
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["start_index"] = entry.start_index
                output["end_index"] = entry.end_index
                output["orig_doc_start"] = entry.orig_doc_start
                output["orig_doc_end"] = entry.orig_doc_end
                nbest_json[kk].append(output)

        assert len(nbest_json['short']) >= 1
        assert len(nbest_json['long']) >= 1

        # We use the [CLS] score of the crop that has the maximum positive score
        # long_score_diff = min_long_score_null - long_best_non_null.start_logit
        # Predict "" if null score - the score of best non-null > threshold
        try:
            crop_unique_id = crops[short_best_non_null.crop_index].unique_id
            start_score_null = unique_id_to_result[crop_unique_id].start_logits[CLS_INDEX]
            end_score_null = unique_id_to_result[crop_unique_id].end_logits[CLS_INDEX]
            short_score_null = start_score_null + end_score_null
            short_score_diff = short_score_null - (short_best_non_null.start_logit +  # bbbbbbbbbbbbbb
                                                   short_best_non_null.end_logit)

            # print(f"short_score: {short_score_diff} and short_null: {short_null_score_diff}")
            if short_score_diff > short_null_score_diff:
                if method == 'restoring' or method == 'mixed':
                    if short_best_non_null.text in long_best_non_null.text:
                        final_pred = (short_best_non_null.text, short_best_non_null.orig_doc_start,
                                      # trick ma capita che non sia dentro la long come token
                                      short_best_non_null.orig_doc_end)
                    else:
                        final_pred = ("", -1, -1)
                        short_num_empty += 1
                else:
                    final_pred = ("", -1, -1)
                    short_num_empty += 1
            else:
                final_pred = (short_best_non_null.text, short_best_non_null.orig_doc_start,
                              short_best_non_null.orig_doc_end)
        except Exception as e:
            final_pred = ("", -1, -1)
            short_num_empty += 1
        try:
            long_score_null = unique_id_to_result[crops[
                long_best_non_null.crop_index].unique_id].long_logits[CLS_INDEX]
            long_score_diff = long_score_null - long_best_non_null.start_logit
            scores_diff_json[example.qas_id] = {'short_score_diff': short_score_diff,
                                                'long_score_diff': long_score_diff}
            # print(f"long_score: {long_score_diff} and long_null: {long_null_score_diff}")

            if long_score_diff > long_null_score_diff:
                final_pred += ("", -1, -1)
                long_num_empty += 1

            else:
                final_pred += (
                    long_best_non_null.text, long_best_non_null.orig_doc_start, long_best_non_null.orig_doc_end)

        except Exception as e:
            print(e)
            final_pred += ("", -1, -1)
            long_num_empty += 1
        all_predictions[example.qas_id] = final_pred
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=2))

    if output_nbest_file is not None:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=2))

    if output_null_log_odds_file is not None:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=2))

    print(f'{short_num_empty} short empty and {long_num_empty} long empty of'
          f' {example_index}')
    return all_predictions


def convert_preds_to_df(preds, candidates, question, true_endToken):
    num_found_long, num_searched_long, num_searched_short, found_end = 0, 0, 0, 0
    df = {'example_id': [], 'PredictionString': [], 'question': [], 'answer': []}
    for example_id, pred in preds.items():
        short_text, start_token, end_token, long_text, long_token, long_token_end = pred
        df['example_id'].append(example_id + '_short')
        df['question'].append(question[example_id])

        short_answer = ''
        if start_token != -1:
            # +1 is required to make the token inclusive
            short_answer = f'{start_token}:{end_token + 1}'
            num_searched_short += 1

        df['PredictionString'].append(short_answer)
        df['answer'].append(short_text)

        # print(entry['document_text'].split(' ')[start_token: end_token + 1])
        # find the long answer
        long_answer = ''
        found_long = False

        min_dist = 1_000_000
        if long_token != -1:
            num_searched_long += 1
            for candidate in candidates[example_id]:
                cstart, cend = candidate.start_token, candidate.end_token
                dist = abs(cstart - long_token)
                if dist < min_dist:
                    min_dist = dist
                if long_token == cstart:
                    if true_endToken:
                        long_answer = f'{long_token}:{long_token_end}'
                    else:
                        long_answer = f'{cstart}:{cend}'
                    found_long = True
                    if long_token_end == cend:
                        found_end += 1
                    break

            if found_long:
                num_found_long += 1
            else:
                logger.info(f"Not found: {min_dist}")

        df['example_id'].append(example_id + '_long')
        df['PredictionString'].append(long_answer)
        df['question'].append(question[example_id])
        df['answer'].append(long_text)

    df = pd.DataFrame(df)
    print(df.astype(bool).sum(axis=0))
    print(f" end uguali {found_end}")
    print(f"Number of long question answered: {num_searched_long} \n Number of long correct answers: {num_found_long}"
          f"\n Number of total questions {len(preds)} \n Number of short answers: {num_searched_short}")
    return df


def enumerate_tags(text_split):
    """Reproduce the preprocessing from:
    A BERT Baseline for the Natural Questions (https://arxiv.org/pdf/1901.08634.pdf)
    We introduce special markup tokens in the doc-ument  to  give  the  model
    a  notion  of  which  partof the document it is reading.  The special
    tokenswe introduced are of the form “[Paragraph=N]”,“[Table=N]”, and “[List=N]”
    at the beginning ofthe N-th paragraph,  list and table respectively
    inthe document. This decision was based on the ob-servation that the first
    few paragraphs and tables inthe document are much more likely than the rest
    ofthe document to contain the annotated answer andso the model could benefit
    from knowing whetherit is processing one of these passages.
    We deviate as follows: Tokens are only created for the first 10 times. All other
    tokens are the same. We only add `special_tokens`. These two are added as they
    make 72.9% + 19.0% = 91.9% of long answers.
    (https://github.com/google-research-datasets/natural-questions)
    """
    special_tokens = ['<P>', '<Table>']
    special_token_counts = [0 for _ in range(len(special_tokens))]
    for index, token in enumerate(text_split):
        for special_token_index, special_token in enumerate(special_tokens):
            if token == special_token:
                cnt = special_token_counts[special_token_index]
                if cnt <= 10:
                    text_split[index] = f'<{special_token[1: -1]}{cnt}>'
                special_token_counts[special_token_index] = cnt + 1

    return text_split


def convert_nq_to_squad(verbose, is_train, args=None):
    """
    Converte il jsonl in un formato compatibile con il Q&A Squad.
    Squad non contiene YES/NO answer(certo 95%)
    :param is_train: flag che indica se siamo in fase di training o evaluation. Training di default
    :param verbose: flag per stampare o meno varei informazioni durante le trasformazioni(solo la prima)
    :param args: None di Default, attualmente passati tramite la funzione getParams1(da cambiare?)
    :return: list of entries
    """
    np.random.seed(123)
    # if args is None:
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--fn', type=str, default='simplified-nq-train.jsonl')
    #     parser.add_argument('--version', type=str, default='v1.0.2')
    #     parser.add_argument('--prefix', type=str, default='nq')
    #     parser.add_argument('--p_val', type=float, default=0.1)
    #     parser.add_argument('--crop_len', type=int, default=2_500)
    #     parser.add_argument('--num_samples', type=int, default=1_000_000)
    #     parser.add_argument('--val_ids', type=str, default='val_ids.csv')
    #     parser.add_argument('--do_enumerate', action='store_true')
    #     parser.add_argument('--do_not_dump', action='store_true')
    #     parser.add_argument('--num_max_tokens', type=int, default=400_000)
    #     args = parser.parse_args()

    if is_train:
        train_fn = f'{args.prefix}-train-{args.version}.json'
        val_fn = f'{args.prefix}-val-{args.version}.json'
        print(f'Converting {args.fn} to {train_fn} & {val_fn} ... ')
    else:
        test_fn = f'{args.prefix}-test-{args.version}.json'
        print(f'Converting {args.fn} to {test_fn} ... ')
    if args.val_ids:
        val_ids = set(str(x) for x in pd.read_csv(args.val_ids)['val_ids'].values)
    else:
        val_ids = set()

    entries = []
    smooth = 0.999
    total_split_len, long_split_len = 0., 0.
    long_end = 0.
    num_very_long, num_yes_no, num_short_dropped, num_trimmed = 0, 0, 0, 0
    num_short_possible, num_long_possible = 0, 0
    max_end_token = -1
    orig_data = {}
    with open(args.fn) as f:
        progress = tqdm(f, total=args.num_samples)
        entry = {}
        for kk, line in enumerate(progress):

            if kk >= args.num_samples:
                break

            data = json.loads(line)
            data_cpy = data.copy()
            example_id = str(data_cpy.pop('example_id'))
            data_cpy['document_text'] = ''
            orig_data[example_id] = data_cpy

            url = 'MISSING' if not is_train else data['document_url']
            # progress.write(f'############ {url} ###############')
            document_text = data['document_text']
            document_text_split = document_text.split(' ')
            if verbose: print("DOCUMENT TEXT AND SPLIT")
            # if verbose: print(document_text)
            # if verbose: print(document_text_split)
            # trim super long
            if len(document_text_split) > args.num_max_tokens:
                num_trimmed += 1
                document_text_split = document_text_split[:args.num_max_tokens]

            if args.do_enumerate:
                document_text_split = enumerate_tags(document_text_split)
            question = data['question_text']  # + '?'
            if verbose: print("QUESTION")
            if verbose: print(question)
            annotations = [None] if not is_train else data['annotations']
            assert len(annotations) == 1, annotations
            if verbose: print("Annotation: " + str(annotations))

            # User str keys!
            example_id = str(data['example_id'])
            candidates = data['long_answer_candidates']
            if verbose: print("CANDIDATES")
            if verbose: print(candidates)
            if not is_train:
                qa = {'question': question, 'id': example_id, 'crop_start': 0}
                context = ' '.join(document_text_split)

            else:
                if verbose: print("In else TRAIN")
                long_answer = annotations[0]['long_answer']
                if verbose: print("LONG ANSWER")
                if verbose: print(long_answer)
                long_answer_len = long_answer['end_token'] - long_answer['start_token']
                total_split_len = smooth * total_split_len + (1. - smooth) * len(
                    document_text_split)
                long_split_len = smooth * long_split_len + (1. - smooth) * \
                                 long_answer_len
                if long_answer['end_token'] > 0:
                    long_end = smooth * long_end + (1. - smooth) * long_answer['end_token']

                if long_answer['end_token'] > max_end_token:
                    max_end_token = long_answer['end_token']

                progress.set_postfix({'ltotal': int(total_split_len),
                                      'llong': int(long_split_len), 'long_end': round(long_end, 2)})

                short_answers = annotations[0]['short_answers']
                if verbose: print("SHORT ANS")
                if verbose: print(short_answers)
                yes_no_answer = annotations[0]['yes_no_answer']
                if yes_no_answer != 'NONE':
                    # progress.write(f'Skipping yes-no: {yes_no_answer}')
                    num_yes_no += 1
                    continue

                if verbose: print(f'Q: {question}')
                if verbose: print(f'L: {long_answer}')
                long_is_impossible = long_answer['start_token'] == -1
                if long_is_impossible:
                    long_answer_candidate = np.random.randint(len(candidates))
                else:
                    long_answer_candidate = long_answer['candidate_index']
                if verbose: print("LONG ANS IMPOSSIBLE STATE:" + str(long_is_impossible))
                long_start_token = candidates[long_answer_candidate]['start_token']
                long_end_token = candidates[long_answer_candidate]['end_token']

                if verbose: print("LONG START: " + str(long_start_token) + "END: " + str(long_end_token))

                # generate crop based on tokens. Note that validation samples should
                # not be cropped as this won't reflect test set performance.
                if args.crop_len > 0 and example_id not in val_ids:
                    crop_start = long_start_token - np.random.randint(int(args.crop_len * 0.75))
                    if verbose: print("IN CROP LEN: " + str(crop_start))

                    if crop_start <= 0:
                        crop_start = 0
                        crop_start_len = -1
                    else:
                        crop_start_len = len(' '.join(document_text_split[:crop_start]))

                    crop_end = crop_start + args.crop_len
                else:
                    crop_start = 0
                    crop_start_len = -1
                    crop_end = 10_000_000

                is_very_long = False
                if long_end_token > crop_end:
                    num_very_long += 1
                    is_very_long = True
                    # progress.write(f'{num_very_long}: Skipping very long answer {long_end_token}, {crop_end}')
                    # continue

                document_text_crop_split = document_text_split[crop_start: crop_end]

                context = ' '.join(document_text_crop_split)
                # create long answer
                long_answers_ = []
                if not long_is_impossible:
                    long_answer_pre_split = document_text_split[:long_answer[
                        'start_token']]
                    long_answer_start = len(' '.join(long_answer_pre_split)) - \
                                        crop_start_len
                    long_answer_split = document_text_split[long_answer['start_token']:
                                                            long_answer['end_token']]
                    long_answer_text = ' '.join(long_answer_split)
                    if not is_very_long:
                        assert context[long_answer_start: long_answer_start + len(
                            long_answer_text)] == long_answer_text, long_answer_text
                    long_answers_ = [{'text': long_answer_text,
                                      'answer_start': long_answer_start}]
                if verbose: print("LONG ANSWER CREATED:" + str(long_answers_))
                # create short answers
                short_is_impossible = len(short_answers) == 0
                short_answers_ = []
                if not short_is_impossible:
                    for short_answer in short_answers:
                        short_start_token = short_answer['start_token']
                        short_end_token = short_answer['end_token']
                        if short_start_token >= crop_start + args.crop_len:
                            num_short_dropped += 1
                            continue
                        short_answers_pre_split = document_text_split[:short_start_token]
                        short_answer_start = len(' '.join(short_answers_pre_split)) - \
                                             crop_start_len
                        short_answer_split = document_text_split[short_start_token: short_end_token]
                        short_answer_text = ' '.join(short_answer_split)
                        assert short_answer_text != ''

                        # this happens if we crop and parts of the short answer overflow
                        short_from_context = context[short_answer_start: short_answer_start + len(short_answer_text)]
                        if short_from_context != short_answer_text:
                            print(f'short diff: {short_from_context} vs {short_answer_text}')
                        short_answers_.append({'text': short_from_context,
                                               'answer_start': short_answer_start})
                if verbose: print("SHORT ANSWER CREATED:" + str(short_answers_))
                if len(short_answers_) == 0:
                    short_is_impossible = True

                if not short_is_impossible:
                    num_short_possible += 1
                if not long_is_impossible:
                    num_long_possible += 1

                qa = {'question': question,
                      'short_answers': short_answers_, 'long_answers': long_answers_,
                      'id': example_id, 'short_is_impossible': short_is_impossible,
                      'long_is_impossible': long_is_impossible,
                      'crop_start': crop_start}
                if verbose: print("QA 1 di Example formato Squad")
                if verbose: print(qa)
            paragraph = {'qas': [qa], 'context': context}
            if verbose: print("PARAGRAFO")
            # if verbose: print(paragraph)
            entry = {'title': url, 'paragraphs': [paragraph]}
            entries.append(entry)
            if verbose: print("ENTRIES")
            # if verbose: print(entries)
            if verbose: verbose = False

    progress.write('  ------------ STATS ------------------')
    progress.write(f'  Number od dropped:  {num_yes_no} yes/no, {num_very_long} very long'
                   f' and {num_short_dropped} short of {kk} and trimmed {num_trimmed}')
    progress.write(f'  #short {num_short_possible} #long {num_long_possible}'
                   f' of {len(entries)}')

    # shuffle to test remaining code
    np.random.shuffle(entries)

    if is_train:
        train_entries, val_entries = [], []
        for entry in entries:
            if entry['paragraphs'][0]['qas'][0]['id'] not in val_ids:
                train_entries.append(entry)
            else:
                val_entries.append(entry)
        if verbose: print("TRAIN ENTRIES: " + str(train_entries))
        if verbose: print("VAL ENTRIES: " + str(val_entries))
        for out_fn, _entries in [(train_fn, train_entries), (val_fn, val_entries)]:
            if not args.do_not_dump:
                with open(out_fn, 'w') as f:
                    json.dump({'version': args.version, 'data': _entries}, f)
                progress.write(f'Wrote {len(_entries)} entries to {out_fn}')

            # save val in competition csv format
            if 'val' in out_fn:

                val_example_ids, val_strs = [], []
                for entry in _entries:
                    example_id = entry['paragraphs'][0]['qas'][0]['id']
                    short_answers = orig_data[example_id]['annotations'][0][
                        'short_answers']
                    sa_str = ''
                    for si, sa in enumerate(short_answers):
                        sa_str += f'{sa["start_token"]}:{sa["end_token"]}'
                        if si < len(short_answers) - 1:
                            sa_str += ' '
                    val_example_ids.append(example_id + '_short')
                    val_strs.append(sa_str)

                    la = orig_data[example_id]['annotations'][0][
                        'long_answer']
                    la_str = ''
                    if la['start_token'] > 0:
                        la_str += f'{la["start_token"]}:{la["end_token"]}'
                    val_example_ids.append(example_id + '_long')
                    val_strs.append(la_str)

                val_df = pd.DataFrame({'example_id': val_example_ids,
                                       'PredictionString': val_strs})
                val_csv_fn = f'{args.prefix}-val-{args.version}.csv'
                val_df.to_csv(val_csv_fn, index=False, columns=['example_id',
                                                                'PredictionString'])
                print(f'Wrote csv to {val_csv_fn}')

    else:  # ELSE di IF training
        if not args.do_not_dump:
            with open(test_fn, 'w') as f:
                json.dump({'version': args.version, 'data': entries}, f)
            progress.write(f'Wrote to {test_fn}')

    if args.val_ids:
        print(f'Using val ids from: {args.val_ids}')
    if verbose: print("PRINT ENTRIES")
    if verbose: print(entries)
    return entries


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'bert_large': (BertConfig, BertTokenizer),
    'albert': (AlbertConfig, AlbertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def get_convert_args(name):
    convert_args = argparse.Namespace()
    convert_args.fn = name
    convert_args.version = 'v0.0.1'
    convert_args.prefix = 'nq'
    convert_args.num_samples = 1_000_000
    convert_args.val_ids = None
    convert_args.do_enumerate = False
    convert_args.do_not_dump = True
    convert_args.num_max_tokens = 400_000
    return convert_args


def get_convert_args1(namefile, max_num_samples):
    """
    :param namefile: path del file per training/evaluation
    :param max_num_samples: massimo numero di oggetti da prendere in considerazione (1mil Default)
    :return: args
    """
    convert_args = argparse.Namespace()
    convert_args.fn = namefile
    convert_args.version = 'v0.0.2'
    convert_args.prefix = 'nq'
    convert_args.p_val = 0.1
    convert_args.crop_len = 2_500
    convert_args.num_samples = max_num_samples
    convert_args.val_ids = None
    convert_args.do_enumerate = 'store_true'
    convert_args.do_not_dump = 'store_true'
    convert_args.num_max_tokens = 400_000

    return convert_args


def toMatrixTensor(crop_or_position, len):
    """
    Genera il target per ogni elemento del crop nel giusto formato, compatibile con il modello
        [yes,no,short,long,no_answer] or [0...1...0]
    :param crop_or_position: una intera entry del crop oppure direttamente start_position o end_position
    :param len: dimensione del vettore da restituire
    :return: singolo vettore Target corrispondente ad una entry
    """

    m = [0] * len
    if len == 5:
        if crop_or_position.long_is_impossible and crop_or_position.short_is_impossible:
            m[4] = 1
        elif crop_or_position.long_is_impossible and not crop_or_position.short_is_impossible:
            m[0] = 1
        elif not crop_or_position.long_is_impossible:
            m[1] = 1
    else:
        m[crop_or_position] = 1
    return tf.convert_to_tensor(m, dtype=tf.int32)


def load_and_cache_crops(args, tokenizer, namefile, verbose, evaluate, max_num_samples, do_cache=False):
    """
    Load data crops from cache or dataset file
    :param do_cache:
    :param max_num_samples:
    :param args: variabili varie ed eventuali
    :param tokenizer: tokenizer(°-°)
    :param namefile: path del file da prendere in considerazione
    :param verbose: flag per stampare o meno varei informazioni durante le trasformazioni(solo la prima)
    :param evaluate: bool False di default. Se True dataset contiene solo gli Input, altrimenti anche i Target
    :return: lista contenente input e target, lista di crops, lista di entries
    """

    if evaluate:
        args_nq = get_convert_args(namefile)
    else:
        args_nq = get_convert_args1(namefile, max_num_samples)

    cached_folder = '../cache/'
    cached_crops_fn = cached_folder + 'cached_test.pkl'
    if os.path.exists(cached_crops_fn) and do_cache:
        print("Loading crops from cached file %s", cached_crops_fn)
        with open(cached_crops_fn, "rb") as f:
            crops = pickle.load(f)
        entries = convert_nq_to_squad(verbose, args=args_nq, is_train=not evaluate)

    else:
        print("NOT loading crops")
        entries = convert_nq_to_squad(verbose, args=args_nq, is_train=not evaluate)
        examples_gen = read_nq_examples(entries, is_training=not evaluate)
        crops = convert_examples_to_crops(examples_gen=examples_gen,
                                          tokenizer=tokenizer,
                                          max_seq_length=args.max_seq_length,
                                          doc_stride=args.doc_stride,
                                          max_query_length=args.max_query_length,
                                          is_training=not evaluate,
                                          cls_token_segment_id=0,
                                          pad_token_segment_id=0,
                                          p_keep_impossible=args.p_keep_impossible if not evaluate else 1.0)
        if do_cache:
            # if cached_folder does not exist create it
            if not os.path.exists(cached_folder):
                os.makedirs(cached_folder)
            with open(cached_crops_fn, "wb") as f:
                pickle.dump(crops, f)

    # stack
    all_input_ids = tf.stack([c.input_ids for c in crops], 0)
    all_attention_mask = tf.stack([c.attention_mask for c in crops], 0)
    all_token_type_ids = tf.stack([c.token_type_ids for c in crops], 0)
    # all_p_mask = tf.stack([c.p_mask for c in crops], 0)

    # cast `tf.bool`
    all_attention_mask = tf.cast(all_attention_mask, tf.int32)

    # old stuff, ma meglio non eliminare
    # all_p_mask = tf.cast(all_p_mask, tf.int32)
    # all_token_type_ids = tf.cast(all_token_type_ids, tf.int32)
    if evaluate:
        dataset = [all_input_ids, all_attention_mask, all_token_type_ids]

    else:
        '''
        all_start_positions = tf.stack([(f.start_position, args.max_seq_length) for f in crops], 0)
        all_end_positions = tf.stack([(f.end_position, args.max_seq_length) for f in crops], 0)
        all_type = tf.stack([(f.long_position) for f in crops])'''
        all_start_positions = tf.convert_to_tensor([f.start_position for f in crops], dtype=tf.int32)
        all_end_positions = tf.convert_to_tensor([f.end_position for f in crops], dtype=tf.int32)
        all_long_positions = tf.convert_to_tensor([f.long_position for f in crops], dtype=tf.int32)
        dataset = [all_input_ids, all_attention_mask, all_token_type_ids,
                   all_start_positions, all_end_positions, all_long_positions]

    return dataset, crops, entries


def getTokenizedDataset(tokenizer, namefile, verbose, max_num_samples):
    """
    La funzione crea input e target per il modello da allenare
    :param max_num_samples: massimo numero di oggetti da prendere in considerazione (1mil Default)
    :param model_type: tipo del modello da utilizzare per tokenizzare
    :param vocab: path del vocabolario del modello
    :param do_lower_case: flag sfigato
    :param namefile: path del file da tokenizzare
    :param verbose: flag per stampare o meno varei informazioni durante le trasformazioni(solo la prima)
    :return: dizionario degli input x e lista dei target y
    """
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument('--short_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument('--long_null_score_diff_threshold', type=float, default=0.0)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=256, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--per_tpu_eval_batch_size", default=4, type=int)
    parser.add_argument("--n_best_size", default=5, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p_keep_impossible', type=float,
                        default=0.03, help="The fraction of impossible"
                                           " samples to keep.")
    parser.add_argument('--do_enumerate', action='store_true')

    args, _ = parser.parse_known_args()
    """
    print(model_type)
    _, tokenizer_class = MODEL_CLASSES[model_type]
    if model_type != "albert_squad":
        tokenizer_class = tokenizer_class(vocab, do_lower_case=do_lower_case)
        tags = get_add_tokens(do_enumerate=args.do_enumerate)
        # num_added = tokenizer_class.add_tokens(tags)
        # print(f"Added {num_added} tokens")
    """
    eval_dataset, crops, entries = load_and_cache_crops(args, tokenizer, namefile, verbose, False,
                                                        max_num_samples)

    do = False
    if do:
        eval_dataset_length = len(eval_dataset[0])
        padded_length = math.ceil(eval_dataset_length / 4) * 4
        num_pad = padded_length - eval_dataset[0].shape[0]
        for ti, t in enumerate(eval_dataset):
            pad_tensor = tf.expand_dims(tf.zeros_like(t[0]), 0)
            pad_tensor = tf.repeat(pad_tensor, num_pad, 0)
            eval_dataset[ti] = tf.concat([t, pad_tensor], 0)

    ret = {
        'input_ids': eval_dataset[0],
        'attention_mask': eval_dataset[1],
        'token_type_ids': eval_dataset[2],
        'start': eval_dataset[3],
        'end': eval_dataset[4],
        'long': eval_dataset[5],
    }
    return ret


def getDatasetForEvaluation(args, tokenizer, namefile, verbose, max_num_samples, do_cache):
    # all_input_ids, all_attention_mask, all_token_type_ids, all_p_mask
    eval_dataset, crops, entries = load_and_cache_crops(args, tokenizer, namefile, verbose, True, max_num_samples,
                                                        do_cache)
    args.eval_batch_size = args.per_tpu_eval_batch_size

    # pad dataset to multiple of `args.eval_batch_size`
    eval_dataset_length = len(eval_dataset[0])
    padded_length = math.ceil(eval_dataset_length / args.eval_batch_size) * args.eval_batch_size
    num_pad = padded_length - eval_dataset[0].shape[0]
    for ti, t in enumerate(eval_dataset):
        pad_tensor = tf.expand_dims(tf.zeros_like(t[0]), 0)
        pad_tensor = tf.repeat(pad_tensor, num_pad, 0)
        eval_dataset[ti] = tf.concat([t, pad_tensor], 0)

    # create eval dataset
    eval_ds = tf.data.Dataset.from_tensor_slices({
        'input_ids': tf.constant(eval_dataset[0]),
        'attention_mask': tf.constant(eval_dataset[1]),
        'token_type_ids': tf.constant(eval_dataset[2]),
        'example_index': tf.range(padded_length, dtype=tf.int32)

    })
    eval_ds = eval_ds.batch(batch_size=args.eval_batch_size, drop_remainder=True)
    # eval_ds = eval_ds.prefetch(tf.data.experimental.AUTOTUNE)
    # eval_ds = strategy.experimental_distribute_dataset(eval_ds)

    # eval
    print("  Num examples = %d", eval_dataset_length)
    print("  Batch size = %d", args.eval_batch_size)

    return eval_ds, crops, entries, eval_dataset_length


def getResult(args, model, eval_ds, crops, entries, eval_dataset_length, do_cache, namefile, app=False):
    csv_fn = '../Results/submission' + args.eval_method + '2.csv'
    padded_length = math.ceil(eval_dataset_length / args.eval_batch_size) * args.eval_batch_size

    @tf.function
    def predict_step(batch):
        outputs = model(batch, training=False)
        return outputs

    all_results = []
    tic = time.time()
    cached_results = '../cache/results_test' + args.eval_method + '.pkl'
    if os.path.exists(cached_results) and do_cache:
        print("Loading results from cached file ", cached_results)
        with open(cached_results, "rb") as f:
            all_results = pickle.load(f)
    else:
        print("NOT loading results")
        for batch_ind, batch in tqdm(enumerate(eval_ds), total=padded_length // args.per_tpu_eval_batch_size):
            # if batch_ind > 2:
            #     break
            example_indexes = batch['example_index']
            # outputs = strategy.experimental_run_v2(predict_step, args=(batch, ))
            outputs = predict_step(batch)
            batched_start_logits = outputs['start'].numpy()
            batched_end_logits = outputs['end'].numpy()
            batched_long_logits = outputs['long'].numpy()

            for i, example_index in enumerate(example_indexes):
                # filter out padded samples
                if example_index >= eval_dataset_length:
                    continue

                eval_crop = crops[example_index]
                unique_id = int(eval_crop.unique_id)
                start_logits = batched_start_logits[i].tolist()
                end_logits = batched_end_logits[i].tolist()
                long_logits = batched_long_logits[i].tolist()

                result = RawResult(unique_id=unique_id,
                                   start_logits=start_logits,
                                   end_logits=end_logits,
                                   long_logits=long_logits)
                all_results.append(result)
        if do_cache:
            with open(cached_results, "wb") as f:
                pickle.dump(all_results, f)

    eval_time = time.time() - tic
    print("  Evaluation done in total %f secs (%f sec per example)",
          eval_time, eval_time / padded_length)
    examples_gen = read_nq_examples(entries, is_training=False)
    preds = write_predictions(examples_gen, crops, all_results, args.n_best_size,
                              args.max_answer_length,
                              None, None, None,
                              args.verbose_logging,
                              args.short_null_score_diff_threshold, args.long_null_score_diff_threshold,
                              args.eval_method)
    del crops, all_results
    gc.collect()
    if not app:
        candidates, question = read_candidates([namefile], do_cache=False)
        sub = convert_preds_to_df(preds, candidates, question, args.true_endToken).sort_values('example_id')
        sub.to_csv(csv_fn, index=False, columns=['example_id', 'PredictionString', 'question', 'answer'])
        print(f'***** Wrote submission to {csv_fn} *****')
        return preds
    else:
        return preds
