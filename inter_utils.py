from __future__ import print_function
from data_utils import _PAD_,_UNK_,_ROOT_,_NUM_
import math
import numpy as np
import random
import sys
from transformers import *

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

def log(*args, **kwargs):
    print(*args,file=sys.stderr, **kwargs)

def pad_batch(batch_data, batch_size, pad_int):
    if len(batch_data) < batch_size:
        batch_data += [[pad_int]] * (batch_size - len(batch_data))
    max_length = max([len(item) for item in batch_data])
    return [item + [pad_int]*(max_length-len(item)) for item in batch_data]

def convert_example_to_features(tokens, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    def convert_tokens_to_ids(tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(tokenizer.vocab.get(token, tokenizer.vocab['[UNK]']))

            #ids.append(tokenizer.vocab[token])
        return ids

    assert isinstance(tokens, list)

    input_tokens = []
    val_pos = []
    end_idx = 0
    for word in tokens:
        #log(word)
        if word == '<NUM>':
            word = str(end_idx)
        b_token = tokenizer.tokenize(word)  # we expect |token| = 1
        #b_token = word
        input_tokens.extend(b_token)
        val_pos.append(end_idx)
        end_idx += len(b_token)

    input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = convert_tokens_to_ids(input_tokens)

    assert len(input_ids) == len(input_tokens)
    assert max(input_ids) < len(tokenizer.vocab)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    return {
        'input_len': len(tokens),
        'input_tokens': input_tokens,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'out_positions': val_pos
    }


## need for input: word, is_Predicate
def get_batch(input_data, batch_size, word2idx, fr_word2idx, lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
              deprel2idx, argument2idx, idx2word, shuffle=False, lang="En", use_bert=False):


    role_number = len(argument2idx)
    #log("start get batch")

    if shuffle and False:
        random.shuffle(input_data)

    for batch_i in range(int(math.ceil(len(input_data)/batch_size))):
        
        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        data_batch = input_data[start_i:end_i]

        role_index_batch = np.zeros((batch_size, role_number), dtype=int)
        role_mask_batch = np.zeros((batch_size, role_number), dtype=int)

        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]
        index_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
        pad_id_batch = np.array(pad_batch(id_batch, batch_size, 0), dtype=int)

        seq_len_batch = [len(sentence) for sentence in data_batch]

        flag_batch = [[int(item[5]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0),dtype=int)

        ## denotes all predicates in sens
        sentence_flags_batch = [[int(item[16])+1 for item in sentence] for sentence in data_batch]
        pad_sentence_flags_batch = np.array(pad_batch(sentence_flags_batch, batch_size, 0),dtype=int)
        flat_flags_batch = np.array([item for line in pad_sentence_flags_batch for item in line])



        predicates_idx_batch = [0]* batch_size
        idx = 0
        for sentence in data_batch:
            for id, item in enumerate(sentence):
                if int(item[5]) == 1:
                    predicates_idx_batch[idx] = id
                    idx += 1
                    break

        text_batch = [[item[6] for item in sentence] for sentence in data_batch]
        if len(text_batch) < batch_size:
            text_batch += [[_PAD_]] * (batch_size - len(text_batch))

        if lang=='En':
            word_batch = [[word2idx.get(item[6],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
            pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))
        else:
            word_batch = [[fr_word2idx.get(item[6], fr_word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
            pad_word_batch = np.array(pad_batch(word_batch, batch_size, fr_word2idx[_PAD_]))

        if use_bert:
            bert_inst_batch = []
            for sen in text_batch:
                bert_inst_batch.append(convert_example_to_features(sen, tokenizer))
            bert_max_length = max([len(inst['input_ids']) for inst in bert_inst_batch])
            batch_length = len(pad_word_batch[0])
            bert_inputs_ids = np.zeros([batch_size, bert_max_length], dtype=np.int64)
            bert_input_mask = np.zeros([batch_size, bert_max_length], dtype=np.int64)
            bert_out_positions = np.empty([batch_size, batch_length], dtype=np.int64)
            for i in range(batch_size):
                berts = bert_inst_batch[i]
                bert_inputs_ids[i, :len(berts['input_ids'])] = berts['input_ids']
                bert_input_mask[i, :len(berts['input_mask'])] = berts['input_mask']
                required_pad = batch_length - len(berts['out_positions'])
                if required_pad > 0:
                    low = berts['out_positions'][-1]
                    assert (bert_max_length - 2) > low
                    bert_out_positions[i] = berts['out_positions'] + [low] * required_pad
                else:
                    bert_out_positions[i] = berts['out_positions']



        else:
            bert_inputs_ids = None
            bert_input_mask = None
            bert_out_positions = None
        word_times_batch = []
        for sentence in data_batch:
            word_dict = dict()
            word_times = []
            for item in sentence:
                word = item[6]
                if word in word_dict:
                    word_dict[word] +=1
                    word_times.append(word_dict[word])
                else:
                    word_dict[word] = 1
                    word_times.append(1)
            word_times_batch.append(word_times)
        pad_word_times_batch = np.array(pad_batch(word_times_batch, batch_size, 0))


        argument_batch = [[argument2idx.get(item[12],argument2idx['<PAD>']) for item in sentence] for sentence in data_batch]
        pad_argument_batch = np.array(pad_batch(argument_batch, batch_size, argument2idx['<PAD>']))
        flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

        if False:
            for i in range(batch_size):
                for j in range(len(data_batch[i])):
                    role = data_batch[i][j][12]
                    role_idx = argument2idx.get(role, argument2idx["_"])
                    if role_idx == 1:
                        continue
                    role_index_batch[i][role_idx] = j
                    role_mask_batch[i][role_idx] = 1
        else:
            role_index_batch = None
            role_mask_batch= None

        if lang=='En':
            pretrain_word_batch = [[pretrain2idx.get(item[6], pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
            pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))
        else:
            pretrain_word_batch = [[fr_pretrain2idx.get(item[6], fr_pretrain2idx[_UNK_]) for item in sentence] for sentence in
                                   data_batch]
            pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

        # flag indicies
        pad_flag_indices = [0 for _ in range(batch_size)]
        for idx in range(batch_size):
            for jdx in range(pad_flag_batch.shape[1]):
                if int(pad_flag_batch[idx, jdx]) == 1:
                    pad_flag_indices[idx] = jdx

        batch = {
            "sentence_id":sentence_id_batch,
            "predicate_id":predicate_id_batch,
            #"fr_predicates_idx":fr_preidx_batch,
            "predicates_idx":predicates_idx_batch,
            "word_id":pad_id_batch,
            "index":index_batch,
            "flag":pad_flag_batch,
            "sen_flags":pad_sentence_flags_batch,
            "flat_flags":flat_flags_batch,
            "word_times":pad_word_times_batch,
            #"fr_flag": fr_pad_flag_batch,
            #"fr_loss_mask":fr_loss_mask_batch,
            "word":pad_word_batch,
            #"fr_word": fr_pad_word_batch,
            #"lemma":pad_lemma_batch,
            #"pos":pad_pos_batch,
            "pretrain":pad_pretrain_word_batch,
            #"fr_pretrain": fr_pad_pretrain_word_batch,
            #"head":pad_head_batch,
            #"rhead":pad_rhead_batch,
            #"deprel":pad_deprel_batch,
            "argument":pad_argument_batch,
            "flat_argument":flat_argument_batch,
            "batch_size":pad_argument_batch.shape[0],
            "pad_seq_len":pad_argument_batch.shape[1],
            "text":text_batch,
            "sentence_len":setence_len_batch,
            "seq_len":seq_len_batch,
            "origin":data_batch,
            'flag_indices':pad_flag_indices,
            #'gold_pos':pad_gold_pos_batch,
            #'gold_head':pad_gold_head_batch,
            #'gold_deprel':pad_gold_deprel_batch,
            'predicates_flag':pad_sentence_flags_batch,
            #'sep_dep_rel': sep_pad_gold_deprel_batch,
            #'sep_dep_link': sep_pad_gold_link_batch,
            'role_index': role_index_batch,
            'role_mask': role_mask_batch,
            'bert_input_ids': bert_inputs_ids,
            'bert_input_mask': bert_input_mask,
            'bert_out_positions': bert_out_positions
        }

        yield batch
            

