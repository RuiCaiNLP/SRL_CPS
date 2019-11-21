import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
#from tree import Tree
from data_utils import _PAD_,_UNK_,_ROOT_,_NUM_

USE_CUDA = torch.cuda.is_available() and True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_torch_variable_from_np(v):
    if USE_CUDA:
        return Variable(torch.from_numpy(v)).cuda()
    else:
        return Variable(torch.from_numpy(v))

def get_torch_variable_from_tensor(t):
    if USE_CUDA:
        return Variable(t).cuda()
    else:
        return Variable(t)

def get_data(v):
    if USE_CUDA:
        return v.data.cpu().numpy()
    else:
        return v.data.numpy()

def create_trees(sentence, deprel2idx):
    ids = [int(item[4]) for item in sentence]
    parents = [int(item[10]) for item in sentence]

    trees = dict()
    roots = dict()

    for i in range(len(ids)):
        tree = Tree(i, deprel2idx.get(sentence[i][11],deprel2idx[_UNK_]), sentence[i][8])
        trees[ids[i]] = tree
    
    for i in range(len(parents)):
        index = ids[i]
        parent = parents[i]
        if parent == 0:
            roots[i] = trees[index]
            continue
        trees[parent].add_child(trees[index])
    
    return trees,roots

def bilinear(x, W, y, input_size, x_seq_len, y_seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    if bias_x:
        bias_one = torch.ones((batch_size, x_seq_len, 1)).to(device)
        x = torch.cat((x, Variable(bias_one)), 2)
    if bias_y:
        bias_one = torch.ones((batch_size, 1)).to(device)
        y = torch.cat((y, Variable(bias_one)), 1)
    nx, ny = input_size + bias_x, input_size + bias_y

    left_part = torch.mm(x.view(batch_size * x_seq_len, -1), W)
    left_part = left_part.view(batch_size, x_seq_len * num_outputs, -1)
    y = y.view(batch_size, -1, 1)
    tag_space = torch.bmm(left_part, y).view(
        batch_size, x_seq_len, num_outputs)

    return tag_space