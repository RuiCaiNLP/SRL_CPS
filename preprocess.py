import os
from data_utils import *


def make_dataset():
    base_path = os.path.join(os.path.dirname(__file__), 'data/CoNLL-2009-Datasets')
    raw_train_file = os.path.join(base_path, 'CoNLL2009-ST-English-train.txt')
    raw_dev_file = os.path.join(base_path, 'CoNLL2009-ST-English-development.txt')
    raw_train_file_fr = os.path.join(base_path, 'FR.TrainSet')
    raw_dev_file_fr = os.path.join(base_path, 'FR.DevSet')
    raw_unlabeled_file_en = os.path.join(os.path.dirname(__file__), 'data/Unlabeled_En.PI')
    raw_unlabeled_file_fr = os.path.join(os.path.dirname(__file__), 'data/Unlabeled_fr.PI')



    train_file = os.path.join(os.path.dirname(__file__), 'data/En_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__), 'data/En_dev.dataset')
    train_file_fr = os.path.join(os.path.dirname(__file__), 'data/Fr_train.dataset')
    dev_file_fr = os.path.join(os.path.dirname(__file__), 'data/Fr_dev.dataset')
    unlabeled_file_en = os.path.join(os.path.dirname(__file__), 'data/Unlabeled_En.dataset')
    unlabeled_file_fr = os.path.join(os.path.dirname(__file__), 'data/Unlabeled_fr.dataset')
    #test_file = os.path.join(os.path.dirname(__file__), 'data/conll09-english/conll09_test.dataset')
    #test_ood_file = os.path.join(os.path.dirname(__file__), 'data/conll09-english/conll09_test_ood.dataset')

    # for train
    with open(raw_train_file, 'r') as fin:
        with open(train_file, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for dev
    with open(raw_dev_file, 'r') as fin:
        with open(dev_file, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for train
    with open(raw_train_file_fr, 'r') as fin:
        with open(train_file_fr, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for dev
    with open(raw_dev_file_fr, 'r') as fin:
        with open(dev_file_fr, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for unlabeled en
    with open(raw_unlabeled_file_en, 'r') as fin:
        with open(unlabeled_file_en, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)

    # for unlabeled en
    with open(raw_unlabeled_file_fr, 'r') as fin:
        with open(unlabeled_file_fr, 'w') as fout:
            while True:
                line = fin.readline()
                if len(line) == 0:
                    break
                fout.write(line)


def stat_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = f.readlines()

        # read data
        sentence_data = []
        sentence = []
        for line in data:
            if len(line.strip()) > 0:
                line = line.strip().split('\t')
                sentence.append(line)
            else:
                sentence_data.append(sentence)
                sentence = []

    predicate_number = 0
    non_predicate_number = 0
    argument_number = 0
    non_argument_number = 0
    predicate_dismatch = 0
    uas_correct = 0
    las_correct = 0
    syntactic_sum = 0
    for sentence in sentence_data:
        for item in sentence:
            syntactic_sum += 1
            if item[8] == item[9]:
                uas_correct += 1
            if item[8] == item[9] and item[10] == item[11]:
                las_correct += 1
            if item[12] == 'Y':
                predicate_number += 1
            else:
                non_predicate_number += 1
            if (item[12] == 'Y' and item[12] == '_') or (item[12] == '_' and item[12] != '_'):
                predicate_dismatch += 1
            for i in range(len(item) - 14):
                if item[14 + i] != '_':
                    argument_number += 1
                else:
                    non_argument_number += 1

    # sentence number
    # predicate number
    # argument number
    print(
    '\tsentence:{} \n\tpredicate:{} non predicate:{} predicate dismatch:{} \n\targument:{} non argument:{} \n\tUAS:{:.2f} LAS:{:.2f}'
    .format(len(sentence_data), predicate_number, non_predicate_number, predicate_dismatch, argument_number,
            non_argument_number, uas_correct / syntactic_sum * 100, las_correct / syntactic_sum * 100))


if __name__ == '__main__':
    # make train/dev/test dataset
    make_dataset()

    train_file = os.path.join(os.path.dirname(__file__), 'data/En_train.dataset')
    dev_file = os.path.join(os.path.dirname(__file__), 'data/En_dev.dataset')
    train_file_fr = os.path.join(os.path.dirname(__file__), 'data/Fr_train.dataset')
    dev_file_fr = os.path.join(os.path.dirname(__file__), 'data/Fr_dev.dataset')
    unlabeled_file_en = os.path.join(os.path.dirname(__file__), 'data/Unlabeled_En.dataset')
    unlabeled_file_fr = os.path.join(os.path.dirname(__file__), 'data/Unlabeled_Fr.dataset')

    # make_dataset_input

    make_dataset_input(train_file, os.path.join(os.path.dirname(__file__), 'temp/train.input'), unify_pred=False)
    make_dataset_input(dev_file, os.path.join(os.path.dirname(__file__), 'temp/dev.input'), unify_pred=False)
    make_dataset_input(train_file_fr, os.path.join(os.path.dirname(__file__), 'temp/train_fr.input'), unify_pred=False)
    make_dataset_input(dev_file_fr, os.path.join(os.path.dirname(__file__), 'temp/dev_fr.input'), unify_pred=False)
    make_dataset_input(unlabeled_file_en, os.path.join(os.path.dirname(__file__), 'temp/unlabeled_en.input'), unify_pred=False)
    make_dataset_input(unlabeled_file_fr, os.path.join(os.path.dirname(__file__), 'temp/unlabeled_fr.input'),
                       unify_pred=False)


    # make word/pos/lemma/deprel/argument vocab
    print('\n-- making (word/lemma/pos/argument) vocab --')
    vocab_path = os.path.join(os.path.dirname(__file__), 'temp')
    print('word:')
    make_word_vocab(unlabeled_file_en, vocab_path, unify_pred=False)
    print('fr word:')
    fr_make_word_vocab(unlabeled_file_fr, vocab_path, unify_pred=False)
    print('pos:')
    make_pos_vocab(train_file, vocab_path, unify_pred=False)
    print('lemma:')
    make_lemma_vocab(train_file, vocab_path, unify_pred=False)
    print('deprel:')
    make_deprel_vocab(train_file, vocab_path, unify_pred=False)
    print('argument:')
    make_argument_vocab(dev_file_fr, train_file_fr, None, vocab_path, unify_pred=False)
    #print('predicate:')
    #make_pred_vocab(train_file, dev_file, None, vocab_path)

    pretrain_path = os.path.join(os.path.dirname(__file__), 'temp')
    deprel_vocab = load_deprel_vocab(os.path.join(pretrain_path, 'deprel.vocab'))
    # shrink pretrained embeding
    print('\n-- shrink pretrained embeding --')
    pretrain_file = os.path.join(os.path.dirname(__file__), 'data/en.vec.txt')  # words.vector
    pretrained_emb_size = 300

    shrink_pretrained_embedding(train_file, dev_file, unlabeled_file_en, pretrain_file, pretrained_emb_size, pretrain_path)

    print('\n-- shrink french pretrained embeding --')
    pretrain_file_fr = os.path.join(os.path.dirname(__file__), 'data/fr.vec.txt')  # words.vector
    pretrained_emb_size_fr = 300
    pretrain_path_fr = os.path.join(os.path.dirname(__file__), 'temp')
    fr_shrink_pretrained_embedding(train_file_fr, dev_file_fr, unlabeled_file_fr, pretrain_file_fr, pretrained_emb_size_fr, pretrain_path_fr)

    make_dataset_input(train_file, os.path.join(pretrain_path, 'train.input'), unify_pred=False,
                       deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(pretrain_path, 'train.pickle.input'))
    make_dataset_input(dev_file, os.path.join(pretrain_path, 'dev.input'), unify_pred=False, deprel_vocab=deprel_vocab,
                       pickle_dump_path=os.path.join(pretrain_path, 'dev.pickle.input'))

    make_dataset_input(train_file_fr, os.path.join(pretrain_path, 'train_fr.input'), unify_pred=False,
                       deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(pretrain_path, 'train_fr.pickle.input'))
    make_dataset_input(dev_file_fr, os.path.join(pretrain_path, 'dev_fr.input'), unify_pred=False, deprel_vocab=deprel_vocab,
                       pickle_dump_path=os.path.join(pretrain_path, 'dev_fr.pickle.input'))

    make_dataset_input(unlabeled_file_en, os.path.join(pretrain_path, 'unlabeled_en.input'), unify_pred=False, deprel_vocab=deprel_vocab,
                       pickle_dump_path=os.path.join(pretrain_path, 'unlabeled_en.pickle.input'))
    make_dataset_input(unlabeled_file_fr, os.path.join(pretrain_path, 'unlabeled_fr.input'), unify_pred=False,
                       deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(pretrain_path, 'unlabeled_fr.pickle.input'))

    log(' data preprocessing finished!')


