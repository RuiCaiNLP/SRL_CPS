#coding=utf-8

import numpy as np

def is_scientific_notation(s):
    s = str(s)
    if s.count(',')>=1:
        sl = s.split(',')
        for item in sl:
            if not item.isdigit():
                return False
        return True
    return False

def is_float(s):
    s = str(s)
    if s.count('.')==1:
        sl = s.split('.')
        left = sl[0]
        right = sl[1]
        if left.startswith('-') and left.count('-')==1 and right.isdigit():
            lleft = left.split('-')[1]
            if lleft.isdigit() or is_scientific_notation(lleft):
                return True
        elif (left.isdigit() or is_scientific_notation(left)) and right.isdigit():
            return True
    return False

def is_fraction(s):
    s = str(s)
    if s.count('\/')==1:
        sl = s.split('\/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True
    if s.count('/')==1:
        sl = s.split('/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True
    if s[-1]=='%' and len(s)>1:
        return True
    return False

def is_number(s):
    s = str(s)
    if s.isdigit() or is_float(s) or is_fraction(s) or is_scientific_notation(s):
        return True
    else:
        return False

unlabeled_file_en = open("data/Unlabeled_En.dataset",'r')
unlabeled_file_fr = open("data/Unlabeled_Fr.dataset",'r')
predicates_set_en = open("data/dev_argument_87.74.pred",'r')
predicates_set_fr = open("data/dev_argument_93.74.pred",'r')
en_emb = "data/en.vec.txt"
fr_emb = "data/fr.vec.txt"



senset_en = []
senset_en_origin = []
sen_en = []
sen_en_origin = []
word_set = set()
for line in unlabeled_file_en:
    if len(line) < 2:
        senset_en.append(sen_en)
        senset_en_origin.append(sen_en_origin)
        sen_en = []
        sen_en_origin = []
        continue
    word = line.strip().split()[1].lower()
    if word not in word_set:
        word_set.add(word.lower())
    sen_en.append(word)
    sen_en_origin.append(line)


senset_fr = []
senset_fr_origin = []
sen_fr = []
sen_fr_origin = []
word_set_fr = set()
for line in unlabeled_file_fr:
    if len(line) < 2:
        senset_fr.append(sen_fr)
        senset_fr_origin.append(sen_fr_origin)
        sen_fr = []
        sen_fr_origin = []
        continue
    word = line.strip().split()[1].lower()
    if word not in word_set_fr:
        word_set_fr.add(word.lower())
    sen_fr.append(word)
    sen_fr_origin.append(line)

print(len(senset_en))
print(len(senset_fr))

pset_en = [None]*10000
p_en = []
id = 0
sen_num = 0
this_num = -1
for line in predicates_set_en:
    if len(line) < 2:
        pset_en[this_num] = p_en
        p_en = []
        id = 0
        continue

    is_p = line.strip().split()[1]
    this_num = int(line.strip().split()[0])
    if is_p == '2':
        p_en.append(id)
    id+=1

pset_fr = [None]*10000
p_fr = []
id = 0
this_num = -1
for line in predicates_set_fr:
    if len(line) < 2:
        pset_fr[this_num] = p_fr
        p_fr = []
        id = 0
        continue

    is_p = line.strip().split()[1]
    this_num = int(line.strip().split()[0])
    if is_p == '2':
        p_fr.append(id)
    id += 1

print(len(pset_en))
print(len(pset_fr))

pretrained_vocab = []
pretrained_embedding = []
with open(en_emb, 'r') as f:
    for line in f.readlines():
        row = line.split(' ')
        word = row[0].lower()
        if not is_number(word):
            if word in word_set and word not in pretrained_vocab:
                pretrained_vocab.append(word)
                weight = [float(item) for item in row[1:]]
                assert (len(weight) == 300)
                pretrained_embedding.append(weight)

pretrained_embedding = np.array(pretrained_embedding, dtype=float)
pretrained_to_idx = {word: idx for idx, word in enumerate(pretrained_vocab)}
idx_to_pretrained = {idx: word for idx, word in enumerate(pretrained_vocab)}

pretrained_vocab_fr = []
pretrained_embedding_fr = []
with open(fr_emb, 'r') as f:
    for line in f.readlines():
        row = line.split(' ')
        word = row[0].lower()
        if not is_number(word):
            if word in word_set_fr and word not in pretrained_vocab_fr:
                pretrained_vocab_fr.append(word)
                weight = [float(item) for item in row[1:]]
                assert (len(weight) == 300)
                pretrained_embedding_fr.append(weight)

pretrained_embedding_fr = np.array(pretrained_embedding_fr, dtype=float)
pretrained_to_idx_fr = {word: idx for idx, word in enumerate(pretrained_vocab_fr)}
idx_to_pretrained_fr = {idx: word for idx, word in enumerate(pretrained_vocab_fr)}
print("embeddings loaded")


print(pretrained_vocab[0:10])
print(pretrained_vocab_fr[0:10])
def calculate(word_en, word_fr):
    if word_en not in pretrained_vocab:
        return 10000.0
    embedding_en = pretrained_embedding[pretrained_to_idx[word_en]]
    if word_fr not in pretrained_vocab_fr:
        return 10000.0
    embedding_fr = pretrained_embedding_fr[pretrained_to_idx_fr[word_fr]]
    return np.sqrt(np.sum(np.square(embedding_en - embedding_fr)))

pair_set = []
effective_number = 0
for id in range(6000):
    print("*"*80)
    sentence_en = senset_en[id]
    sentence_fr = senset_fr[id]
    len_en = len(sentence_en)
    len_fr = len(sentence_fr)
    if len_en*len_fr == 0:
        pair_set.append(None)
        continue
    if pset_en[id] is None or pset_fr[id] is None:
        pair_set.append(None)
        continue
    distances = np.zeros((len_en, len_fr), dtype="float32")
    for i in range(len_en):
        for j in range(len_fr):
            distances[i][j] = calculate(sentence_en[i].lower(), sentence_fr[j].lower())
    argmin_en = np.argmin(distances, axis=1)
    argmin_fr = np.argmin(distances, axis=0)
    pair = None
    print(id)
    print(sentence_en)
    print(sentence_fr)
    for p_id in pset_en[id]:
        if argmin_en[p_id] in pset_fr[id]:
            if argmin_fr[argmin_en[p_id]] == p_id:
                pair = (p_id, argmin_en[p_id])
                break
    pair_set.append(pair)
    if pair != None:
        print(sentence_en[pair[0]], sentence_fr[pair[1]])
        effective_number += 1



target_file_en = open("data/Unlabeled_En.PI", 'w')
for i in range(len(pair_set)):
    if pair_set[i] == None:
        continue
    for j in range(len(senset_en_origin[i])):
        parts = senset_en_origin[i][j].strip().split()
        if j == 0:
            parts[12] = '_'
        if pair_set[i]!=None:
            if j == pair_set[i][0]:
                parts[12] = 'Y'

        target_file_en.write('\t'.join(parts))

        target_file_en.write('\n')
    target_file_en.write('\n')

target_file_fr = open("data/Unlabeled_fr.PI", 'w')
for i in range(len(pair_set)):
    if pair_set[i] == None:
        continue
    for j in range(len(senset_fr_origin[i])):
        parts = senset_fr_origin[i][j].strip().split()
        if j == 0:
            parts[12] = '_'
        if pair_set[i]!=None:
            if j == pair_set[i][1]:
                parts[12] = 'Y'

        target_file_fr.write('\t'.join(parts))

        target_file_fr.write('\n')
    target_file_fr.write('\n')