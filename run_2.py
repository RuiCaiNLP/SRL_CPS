from __future__ import print_function
import model
import data_utils
import inter_utils
import pickle
import time
import os
import torch
import sys
from torch import nn
from torch import optim
from tqdm import tqdm
import argparse

from utils import USE_CUDA

from utils import get_torch_variable_from_np, get_data
from scorer import eval_train_batch, eval_data
from data_utils import output_predict

from data_utils import *


#def log(*args, **kwargs):
    #print(*args,file=sys.stderr, **kwargs)
#    print(*args,  **kwargs)


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def print_PRF(probs, gold):
    predicts = np.argmax(probs.cpu().data.numpy(), axis=1)
    gold = gold.cpu().data.numpy()
    correct = 0.0
    NonullTruth = 0.0
    NonullPredict = 0.1
    for p, g in zip(predicts, gold):
        if g == 0:
            continue
        if g > 1:
            NonullTruth += 1
        if p > 1:
            NonullPredict += 1
        if p == g and g > 1:
            correct += 1

    P = correct/NonullPredict + 0.0001
    R = correct/NonullTruth
    F = 2*P*R/(P+R)
    print(correct, NonullPredict, NonullTruth)
    print(P, R, F)


def make_parser():
    parser = argparse.ArgumentParser(description='A Unified Syntax-aware SRL model')

    # input
    #parser.add_argument('--train_data', type=str, help='Train Dataset with CoNLL09 format')
    #parser.add_argument('--valid_data', type=str, help='Train Dataset with CoNLL09 format')
    #parser.add_argument('--train_data_fr', type=str, help='Train Dataset with CoNLL09 format')
    #parser.add_argument('--valid_data_fr', type=str, help='Train Dataset with CoNLL09 format')
    #parser.add_argument('--unlabeled_data_en', type=str, help='Train Dataset with CoNLL09 format')
    #parser.add_argument('--unlabeled_data_fr', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--seed', type=int, default=100, help='the random seed')

    # this default value is from PATH LSTM, you can just follow it too
    # if you want to do the predicate disambiguation task, you can replace the accuracy with yours.
    parser.add_argument('--dev_pred_acc', type=float, default=0.9477,
                        help='Dev predicate disambiguation accuracy')
    parser.add_argument('--test_pred_acc', type=float, default=0.9547,
                        help='Test predicate disambiguation accuracy')
    parser.add_argument('--ood_pred_acc', type=float, default=0.8618,
                        help='OOD predicate disambiguation accuracy')

    # preprocess
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess')
    #parser.add_argument('--tmp_path', type=str, help='temporal path')
    #parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--result_path', type=str, help='result path')
    #parser.add_argument('--pretrain_embedding', type=str, help='Pretrain embedding like GloVe or word2vec')
    parser.add_argument('--pretrain_emb_size', type=int, default=100,
                        help='size of pretrain word embeddings')

    # train
    parser.add_argument('--train', action='store_true',
                        help='Train')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Train epochs')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout when training')
    parser.add_argument('--dropout_word', type=float, default=0.3,
                        help='Dropout when training')
    parser.add_argument('--dropout_mlp', type=float, default=0.3,
                        help='Dropout when training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size in train and eval')
    parser.add_argument('--word_emb_size', type=int, default=100,
                        help='Word embedding size')
    parser.add_argument('--pos_emb_size', type=int, default=32,
                        help='POS tag embedding size')
    parser.add_argument('--lemma_emb_size', type=int, default=100,
                        help='Lemma embedding size')

    parser.add_argument('--bilstm_hidden_size', type=int, default=512,
                        help='Bi-LSTM hidden state size')
    parser.add_argument('--bilstm_num_layers', type=int, default=4,
                        help='Bi-LSTM layer number')
    parser.add_argument('--valid_step', type=int, default=1000,
                        help='Valid step size')

    parser.add_argument('--use_deprel', action='store_true',
                        help='[USE] dependency relation')
    parser.add_argument('--deprel_emb_size', type=int, default=64,
                        help='Dependency relation embedding size')

    parser.add_argument('--use_highway', action='store_true',
                        help='[USE] highway connection')
    parser.add_argument('--highway_num_layers', type=int, default=10,
                        help='Highway layer number')

    parser.add_argument('--use_biaffine', action='store_true',
                        help='[USE] highway connection')

    parser.add_argument('--use_self_attn', action='store_true',
                        help='[USE] self attention')
    parser.add_argument('--self_attn_heads', type=int, default=10,
                        help='Self attention Heads')

    parser.add_argument('--use_flag_emb', action='store_true',
                        help='[USE] predicate flag embedding')
    parser.add_argument('--flag_emb_size', type=int, default=16,
                        help='Predicate flag embedding size')

    parser.add_argument('--use_elmo', action='store_true',
                        help='[USE] ELMo embedding')
    parser.add_argument('--elmo_emb_size', type=int, default=300,
                        help='ELMo embedding size')
    parser.add_argument('--elmo_options', type=str,
                        help='ELMo options file')
    parser.add_argument('--elmo_weight', type=str,
                        help='ELMo weight file')

    parser.add_argument('--use_bert', action='store_true',
                        help='[USE] BERT embedding')

    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')

    # eval
    parser.add_argument('--eval', action='store_true',
                        help='Eval')
    parser.add_argument('--model', type=str, help='Model')

    return parser


if __name__ == '__main__':
    print('cross-lingual model')


    args = make_parser().parse_args()

    # set random seed
    seed_everything(args.seed, USE_CUDA)


    use_bert = args.use_bert

    # do preprocessing

    print('\t start loading data...')
    start_t = time.time()

    train_input_file = os.path.join(os.path.dirname(__file__), 'temp/train.pickle.input')
    dev_input_file = os.path.join(os.path.dirname(__file__), 'temp/dev.pickle.input')
    train_data = data_utils.load_dump_data(train_input_file)
    dev_data = data_utils.load_dump_data(dev_input_file)
    train_dataset = train_data['input_data']
    dev_dataset = dev_data['input_data']

    train_input_file_fr = os.path.join(os.path.dirname(__file__), 'temp/train_fr.pickle.input')
    dev_input_file_fr = os.path.join(os.path.dirname(__file__), 'temp/dev_fr.pickle.input')
    train_data_fr = data_utils.load_dump_data(train_input_file_fr)
    dev_data_fr = data_utils.load_dump_data(dev_input_file_fr)
    train_dataset_fr = train_data_fr['input_data']
    dev_dataset_fr = dev_data_fr['input_data']
    #log(len(train_dataset_fr))
    #log(len(dev_dataset_fr))
    labeled_dataset_fr = train_dataset_fr + dev_dataset_fr
    #log(len(labeled_dataset_fr))

    unlabeled_file_en = os.path.join(os.path.dirname(__file__), 'temp/unlabeled_en.pickle.input')
    unlabeled_file_fr = os.path.join(os.path.dirname(__file__), 'temp/unlabeled_fr.pickle.input')
    unlabeled_data_en = data_utils.load_dump_data(unlabeled_file_en)
    unlabeled_data_fr = data_utils.load_dump_data(unlabeled_file_fr)
    unlabeled_dataset_en = unlabeled_data_en['input_data']
    unlabeled_dataset_fr = unlabeled_data_fr['input_data']


    word2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/word2idx.bin'))
    idx2word = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/idx2word.bin'))

    fr_word2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/fr_word2idx.bin'))
    fr_idx2word = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/fr_idx2word.bin'))

    lemma2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/lemma2idx.bin'))
    idx2lemma = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/idx2lemma.bin'))

    pos2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/pos2idx.bin'))
    idx2pos = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/idx2pos.bin'))

    deprel2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/deprel2idx.bin'))
    idx2deprel = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/idx2deprel.bin'))

    pretrain2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/pretrain2idx.bin'))
    idx2pretrain = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/idx2pretrain.bin'))

    fr_pretrain2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/fr_pretrain2idx.bin'))
    fr_idx2pretrain = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/fr_idx2pretrain.bin'))

    argument2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/argument2idx.bin'))
    idx2argument = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/idx2argument.bin'))

    pretrain_emb_weight = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/pretrain.emb.bin'))
    fr_pretrain_emb_weight = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__), 'temp/fr_pretrain.emb.bin'))

    print('\t data loading finished! consuming {} s'.format(int(time.time() - start_t)))

    print(word2idx['<UNK>'])
    print(pretrain_emb_weight[word2idx['<UNK>']])
    print(word2idx['<PAD>'])
    print(pretrain_emb_weight[fr_word2idx['<PAD>']])
    # result_path = os.path.join(os.path.dirname(__file__),'result/')

    result_path = args.result_path

    print('\t start building model...')
    start_t = time.time()

    dev_predicate_sum = dev_data['predicate_sum']
    dev_predicate_correct = int(dev_predicate_sum * args.dev_pred_acc)


    # hyper parameters
    max_epoch = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    dropout = args.dropout
    dropout_word = args.dropout_word
    dropout_mlp = args.dropout_mlp
    use_biaffine = args.use_biaffine
    word_embedding_size = args.word_emb_size
    pos_embedding_size = args.pos_emb_size
    pretrained_embedding_size = args.pretrain_emb_size
    lemma_embedding_size = args.lemma_emb_size

    use_deprel = args.use_deprel
    deprel_embedding_size = args.deprel_emb_size

    bilstm_hidden_size = args.bilstm_hidden_size
    bilstm_num_layers = args.bilstm_num_layers
    show_steps = args.valid_step

    use_highway = args.use_highway
    highway_layers = args.highway_num_layers

    use_flag_embedding = args.use_flag_emb
    flag_embedding_size = args.flag_emb_size

    use_elmo = args.use_elmo
    elmo_embedding_size = args.elmo_emb_size
    elmo_options_file = args.elmo_options
    elmo_weight_file = args.elmo_weight
    elmo = None


    use_self_attn = args.use_self_attn
    self_attn_head = args.self_attn_heads

    if args.train:
        FLAG = 'TRAIN'
    if args.eval:
        FLAG = 'EVAL'
        MODEL_PATH = args.model

    if FLAG == 'TRAIN':
        model_params = {
            "dropout": dropout,
            "dropout_word": dropout_word,
            "dropout_mlp": dropout_mlp,
            "use_biaffine": use_biaffine,
            "batch_size": batch_size,
            "word_vocab_size": len(word2idx),
            "fr_word_vocab_size": len(fr_word2idx),
            "lemma_vocab_size": len(lemma2idx),
            "pos_vocab_size": len(pos2idx),
            "deprel_vocab_size": len(deprel2idx),
            "pretrain_vocab_size": len(pretrain2idx),
            "fr_pretrain_vocab_size": len(fr_pretrain2idx),
            "word_emb_size": word_embedding_size,
            "lemma_emb_size": lemma_embedding_size,
            "pos_emb_size": pos_embedding_size,
            "pretrain_emb_size": pretrained_embedding_size,
            "pretrain_emb_weight": pretrain_emb_weight,
            "fr_pretrain_emb_weight": fr_pretrain_emb_weight,
            "bilstm_num_layers": bilstm_num_layers,
            "bilstm_hidden_size": bilstm_hidden_size,
            "target_vocab_size": len(argument2idx),
            "use_highway": use_highway,
            "highway_layers": highway_layers,
            "use_self_attn": use_self_attn,
            "self_attn_head": self_attn_head,
            "use_deprel": use_deprel,
            "deprel_emb_size": deprel_embedding_size,
            "deprel2idx": deprel2idx,
            "use_flag_embedding": use_flag_embedding,
            "flag_embedding_size": flag_embedding_size,
            'use_elmo': use_elmo,
            "use_bert": use_bert,
            "elmo_embedding_size": elmo_embedding_size,
            "elmo_options_file": elmo_options_file,
            "elmo_weight_file": elmo_weight_file,
            #"use_tree_lstm": use_tree_lstm,
            #"use_gcn": use_gcn,
            #"use_sa_lstm": use_sa_lstm,
            #"use_rcnn": use_rcnn
        }

        # build model
        srl_model = model.SR_Model(model_params)

        if USE_CUDA:
            srl_model.cuda()

        criterion_word = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(srl_model.parameters(), lr=learning_rate)
        optimizer_para = optim.Adam(srl_model.SR_Labeler.parameters(), lr=learning_rate)

        print(srl_model)

        print('\t model build finished! consuming {} s'.format(int(time.time() - start_t)))

        print('\nStart training...')
        dev_best_score = None
        test_best_score = None
        test_ood_best_score = None
        use_bert = True
        unlabeled_Generator_En = inter_utils.get_batch(unlabeled_dataset_en, batch_size, word2idx, fr_word2idx,
                                                       lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
                                                       deprel2idx, argument2idx, idx2word, shuffle=False,
                                                       lang="En", use_bert=use_bert, para=True)
        unlabeled_Generator_Fr = inter_utils.get_batch(unlabeled_dataset_fr, batch_size, word2idx, fr_word2idx,
                                                       lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
                                                       deprel2idx, argument2idx, idx2word, shuffle=False,
                                                       lang="Fr", use_bert=use_bert, para=True)

        for epoch in range(30):

            for batch_i, train_input_data in enumerate(
                    inter_utils.get_batch(train_dataset, batch_size, word2idx, fr_word2idx,
                                          lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
                                          deprel2idx, argument2idx, idx2word, shuffle=True,
                                          lang='En', use_bert=use_bert)):
                srl_model.train()
                flat_argument = train_input_data['flat_argument']
                target_batch_variable = get_torch_variable_from_np(flat_argument)

                out, out_word = srl_model(train_input_data, lang='En', use_bert=use_bert)
                #_,  prediction_batch_variable = torch.max(out, 1)
                loss = criterion(out, target_batch_variable)
                loss_word = criterion_word(out_word, target_batch_variable)
                if batch_i % 50 == 0:
                    print("epoch:", epoch, batch_i, loss.item(), loss_word.item())
                optimizer.zero_grad()
                (loss+loss_word).backward()
                optimizer.step()



                #batch_size=1
                try:
                    unlabeled_data_en = next(unlabeled_Generator_En)
                    unlabeled_data_fr = next(unlabeled_Generator_Fr)
                except StopIteration:
                    unlabeled_Generator_En = inter_utils.get_batch(unlabeled_dataset_en, batch_size, word2idx,
                                                                   fr_word2idx,
                                                                   lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
                                                                   deprel2idx, argument2idx, idx2word, shuffle=False,
                                                                   lang="En", use_bert=True, para=True)
                    unlabeled_Generator_Fr = inter_utils.get_batch(unlabeled_dataset_fr, batch_size, word2idx,
                                                                   fr_word2idx,
                                                                   lemma2idx, pos2idx, pretrain2idx, fr_pretrain2idx,
                                                                   deprel2idx, argument2idx, idx2word, shuffle=False,
                                                                   lang="Fr",use_bert=True, para=True)
                    unlabeled_data_en = next(unlabeled_Generator_En)
                    unlabeled_data_fr = next(unlabeled_Generator_Fr)



                
                u_loss_pair, loss_word = srl_model((unlabeled_data_en, unlabeled_data_fr), lang='En', unlabeled=True,
                                                    self_constrain=False, use_bert=use_bert)
                optimizer.zero_grad()
                u_loss, u_loss_2, coverage = u_loss_pair
                (u_loss + u_loss_2).backward()
                optimizer.step()
                batch_size = 30
                
                if batch_i % 50 == 0:
                    print("para loss:", batch_i, u_loss.item(), u_loss_2.item())
                    #print(coverage)

                if batch_i > 0 and batch_i % show_steps == 0:
                    srl_model.eval()
                    _, pred = torch.max(out, 1)
                    pred = get_data(pred)

                    print('\n')
                    print('*' * 80)

                    #eval_train_batch(epoch, batch_i, loss.item(), flat_argument, pred, argument2idx)

                    print('FR test:')
                    score, dev_output = eval_data(srl_model, elmo, labeled_dataset_fr, batch_size, word2idx,
                                                  fr_word2idx,
                                                  lemma2idx,
                                                  pos2idx, pretrain2idx, fr_pretrain2idx, deprel2idx, argument2idx,
                                                  idx2argument, idx2word,
                                                  False,
                                                  dev_predicate_correct, dev_predicate_sum, lang='Fr', use_bert=use_bert)
                    print('En test:')
                    eval_data(srl_model, elmo, dev_dataset, batch_size, word2idx,
                              fr_word2idx,
                              lemma2idx,
                              pos2idx, pretrain2idx, fr_pretrain2idx, deprel2idx, argument2idx,
                              idx2argument, idx2word,
                              False,
                              dev_predicate_correct, dev_predicate_sum, lang='En', use_bert=use_bert)

                    if dev_best_score is None or score[5] > dev_best_score[5]:
                        dev_best_score = score
                        #output_predict(
                        ##    os.path.join(result_path, 'dev_argument_{:.2f}.pred'.format(dev_best_score[2] * 100)),
                         #   dev_output)
                        # torch.save(srl_model, os.path.join(os.path.dirname(__file__),'model/best_{:.2f}.pkl'.format(dev_best_score[2]*100)))
                    print('\tdev best P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(
                        dev_best_score[0] * 100, dev_best_score[1] * 100,
                        dev_best_score[2] * 100, dev_best_score[3] * 100,
                        dev_best_score[4] * 100, dev_best_score[5] * 100))

    else:

        srl_model = torch.load(MODEL_PATH)
        srl_model.eval()
        log('test not available')
        """
        score, test_output = eval_data(srl_model, elmo, test_dataset, batch_size, word2idx, lemma2idx, pos2idx,
                                       pretrain2idx, deprel2idx, argument2idx, idx2argument, False,
                                       test_predicate_correct, test_predicate_sum)
        log('\ttest P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(score[0] * 100, score[1] * 100,
                                                                                         score[2] * 100, score[3] * 100,
                                                                                         score[4] * 100,
                                                                                         score[5] * 100))

        if test_ood_file is not None:
            log('ood:')
            score, ood_output = eval_data(srl_model, elmo, test_ood_dataset, batch_size, word2idx, lemma2idx, pos2idx,
                                          pretrain2idx, deprel2idx, argument2idx, idx2argument, False,
                                          test_ood_predicate_correct, test_ood_predicate_sum)
            output_predict(os.path.join(result_path, 'ood_argument_{:.2f}.pred'.format(score[2] * 100)), ood_output)
            log(
            '\tood P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(score[0] * 100, score[1] * 100,
                                                                                      score[2] * 100, score[3] * 100,
                                                                                      score[4] * 100, score[5] * 100))
        """

