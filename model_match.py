from __future__ import print_function
import numpy as np
import torch
from torch import nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F

from utils import USE_CUDA
from utils import get_torch_variable_from_np, get_data
from utils import bilinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log(*args, **kwargs):
    print(*args,file=sys.stderr, **kwargs)

class SR_Labeler(nn.Module):
    def __init__(self, model_params):
        super(SR_Labeler, self).__init__()
        self.dropout_word = nn.Dropout(p=0.5)
        self.dropout_hidden = nn.Dropout(p=0.3)
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']

        if USE_CUDA:
            self.bilstm_hidden_state = (
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state = (
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))
        self.bilstm_layer = nn.LSTM(input_size=300 + 1 * self.flag_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers,
                                    bidirectional=True,
                                    bias=True, batch_first=True)


        self.mlp_size = 300
        self.rel_W = nn.Parameter(
            torch.from_numpy(
                np.zeros((self.mlp_size + 1, self.target_vocab_size * (self.mlp_size + 1))).astype("float32")).to(
                device))
        self.mlp_arg = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())
        self.mlp_pred = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size, self.mlp_size), nn.ReLU())

    def forward(self, pretrained_emb, flag_emb, predicates_1D, seq_len, para=False):
        input_emb = torch.cat((pretrained_emb, flag_emb), 2)
        if para == False:
            input_emb = self.dropout_word(input_emb)

        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)
        if para == False:
            hidden_input = self.dropout_hidden(hidden_input)
        arg_hidden = self.mlp_arg(hidden_input)
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.mlp_pred(pred_recur)
        SRL_output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                              num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        SRL_output = SRL_output.view(self.batch_size * seq_len, -1)
        return SRL_output


class SR_Compressor(nn.Module):
    def __init__(self, model_params):
        super(SR_Compressor, self).__init__()
        self.dropout_word = nn.Dropout(p=0.0)
        self.dropout_hidden = nn.Dropout(p=0.0)
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']

        self.hidden2vector = nn.Sequential(nn.Linear(self.pretrain_emb_size+self.flag_emb_size, 300),
                                                nn.ReLU(),
                                                nn.Linear(300, 200))

        self.reProb = nn.Sequential(nn.Linear(self.target_vocab_size*2, self.target_vocab_size),
                                    nn.ReLU(),
                                    nn.Linear(self.target_vocab_size, self.target_vocab_size),
                                    nn.Sigmoid())

    def forward(self, SRL_input, pretrained_emb, word_id_emb, seq_len, para=False):
        SRL_input = SRL_input.view(self.batch_size, seq_len, -1)
        compress_input = torch.cat((pretrained_emb, word_id_emb), 2)
        # B T V
        role_vectors = self.hidden2vector(compress_input)
        # B T R V
        role_vectors = role_vectors.unsqueeze(2).expand(self.batch_size, seq_len, self.target_vocab_size, 200)
        # B T R
        word_weights = F.softmax(SRL_input, dim=2).view(self.batch_size, seq_len, self.target_vocab_size)
        # B T R -> B 1 R ->  B T R
        weights_sum = torch.sum(word_weights, dim=1, keepdim=True).expand(self.batch_size, seq_len, self.target_vocab_size)
        O_weights = weights_sum - word_weights
        all_weights = torch.cat((word_weights, O_weights), 2)
        # B T 2R -> B T
        re_weights = self.reProb(all_weights).view(self.batch_size, seq_len, self.target_vocab_size, 1)
        # B R V
        compressor_vector = torch.sum(role_vectors*re_weights, dim=1)
        return compressor_vector


class SR_Matcher(nn.Module):
    def __init__(self, model_params):
        super(SR_Matcher, self).__init__()
        self.dropout_word = nn.Dropout(p=0.0)
        self.mlp_size = 300
        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.matrix = nn.Parameter(
                    get_torch_variable_from_np(np.zeros((self.pretrain_emb_size+self.flag_emb_size, 200)).astype("float32")))

    def forward(self, role_vectors, pretrained_emb, word_id_emb, seq_len, para=False):
        query_vector = torch.cat((pretrained_emb, word_id_emb), 2)
        role_vectors = role_vectors.view(self.batch_size, self.target_vocab_size, 200)
        # B T R V
        role_vectors = role_vectors.unsqueeze(1).expand(self.batch_size, seq_len, self.target_vocab_size, 200)
        # B T R W
        query_vector = query_vector.unsqueeze(2).expand(self.batch_size, seq_len, self.target_vocab_size,
                                                        self.pretrain_emb_size+self.flag_emb_size)
        # B T R V
        y = torch.mm(query_vector.contiguous().view(self.batch_size*seq_len*self.target_vocab_size, -1), self.matrix)
        # B T R
        y = y.contiguous().view(self.batch_size, seq_len, self.target_vocab_size, 200)
        roles_scores = torch.sum(role_vectors*y, dim=3)
        zerosNull = get_torch_variable_from_np(np.zeros((self.batch_size, seq_len, 1), dtype='float32'))
        roles_scores = roles_scores.view(self.batch_size, seq_len, -1)

        output_word = torch.cat((roles_scores[:,:,0:1], zerosNull, roles_scores[:,:,2:]), 2)
        output_word = output_word.view(self.batch_size * seq_len, -1)

        return output_word




class SR_Model(nn.Module):
    def __init__(self, model_params):
        super(SR_Model, self).__init__()
        self.dropout = model_params['dropout']

        self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.use_biaffine = model_params['use_biaffine']

        self.word_vocab_size = model_params['word_vocab_size']
        self.fr_word_vocab_size = model_params['fr_word_vocab_size']
        self.pretrain_vocab_size = model_params['pretrain_vocab_size']
        self.fr_pretrain_vocab_size = model_params['fr_pretrain_vocab_size']
        self.word_emb_size = model_params['word_emb_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']
        self.pretrain_emb_weight = model_params['pretrain_emb_weight']
        self.fr_pretrain_emb_weight = model_params['fr_pretrain_emb_weight']

        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']

        self.pretrained_embedding = nn.Embedding(self.pretrain_vocab_size, self.pretrain_emb_size)
        self.pretrained_embedding.weight.data.copy_(torch.from_numpy(self.pretrain_emb_weight))
        self.fr_pretrained_embedding = nn.Embedding(self.fr_pretrain_vocab_size, self.pretrain_emb_size)
        self.fr_pretrained_embedding.weight.data.copy_(torch.from_numpy(self.fr_pretrain_emb_weight))

        self.word_matrix = nn.Linear(self.pretrain_emb_size, self.pretrain_emb_size)
        self.word_matrix.weight.data.copy_(torch.from_numpy(np.eye(self.pretrain_emb_size, self.pretrain_emb_size, dtype="float32")))

        self.id_embedding = nn.Embedding(100, self.flag_emb_size)
        self.id_embedding.weight.data.uniform_(-1.0, 1.0)

        if self.use_flag_embedding:
            self.flag_embedding = nn.Embedding(2, self.flag_emb_size)
            self.flag_embedding.weight.data.uniform_(-1.0, 1.0)

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_emb_size)
        self.word_embedding.weight.data.uniform_(-1.0, 1.0)

        self.fr_word_embedding = nn.Embedding(self.fr_word_vocab_size, self.word_emb_size)
        self.fr_word_embedding.weight.data.uniform_(-1.0, 1.0)


        self.word_dropout = nn.Dropout(p=0.5)
        self.out_dropout = nn.Dropout(p=0.3)

        input_emb_size = 0
        if self.use_flag_embedding:
            input_emb_size += self.flag_emb_size
        else:
            input_emb_size += 1
        self.SR_Labeler = SR_Labeler(model_params)
        self.SR_Compressor = SR_Compressor(model_params)
        self.SR_Matcher = SR_Matcher(model_params)



    def parallel_train(self, batch_input):
        unlabeled_data_en, unlabeled_data_fr = batch_input

        pretrain_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['pretrain'])
        predicates_1D_fr = unlabeled_data_fr['predicates_idx']
        flag_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['flag'])
        word_id_fr = get_torch_variable_from_np(unlabeled_data_fr['word_times'])
        word_id_emb_fr = self.id_embedding(word_id_fr).detach()
        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()
        pretrain_emb_fr = self.fr_pretrained_embedding(pretrain_batch_fr).detach()

        pretrain_batch = get_torch_variable_from_np(unlabeled_data_en['pretrain'])
        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        word_id = get_torch_variable_from_np(unlabeled_data_en['word_times'])
        word_id_emb = self.id_embedding(word_id)
        flag_emb = self.flag_embedding(flag_batch)
        seq_len = flag_emb.shape[1]
        seq_len_en = seq_len
        pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()

        seq_len = flag_emb.shape[1]
        SRL_output = self.SR_Labeler(pretrain_emb, flag_emb.detach(), predicates_1D, seq_len, para=True)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        pred_recur = self.SR_Compressor(SRL_input.detach(), pretrain_emb, word_id_emb.detach(), seq_len, para=True)



        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(pretrain_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True)

        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        pred_recur_fr = self.SR_Compressor(SRL_input_fr, pretrain_emb_fr, word_id_emb_fr, seq_len_fr, para=True)

        L2_loss_function = nn.MSELoss(size_average=False)
        l2_loss = L2_loss_function(pred_recur_fr, pred_recur.detach())/self.batch_size
        """
        En event vector, En word
        """
        output_word_en = self.SR_Matcher(pred_recur.detach(), pretrain_emb,  word_id_emb.detach(), seq_len, para=True)

        #############################################
        """
        Fr event vector, En word
        """
        output_word_fr = self.SR_Matcher(pred_recur_fr, pretrain_emb, word_id_emb.detach(), seq_len, para=True)
        unlabeled_loss_function = nn.KLDivLoss(size_average=False)
        output_word_en = F.softmax(output_word_en, dim=1).detach()
        output_word_fr = F.log_softmax(output_word_fr, dim=1)
        loss = unlabeled_loss_function(output_word_fr, output_word_en)/(seq_len_en*self.batch_size)
        #############################################3
        """
        En event vector, Fr word
        """
        output_word_en = self.SR_Matcher(pred_recur.detach(), pretrain_emb_fr, word_id_emb_fr.detach(), seq_len_fr, para=True)
        """
        Fr event vector, Fr word
        """
        output_word_fr = self.SR_Matcher(pred_recur_fr, pretrain_emb_fr, word_id_emb_fr.detach(), seq_len_fr, para=True)
        unlabeled_loss_function = nn.KLDivLoss(size_average=False)
        output_word_en = F.softmax(output_word_en, dim=1).detach()
        output_word_fr = F.log_softmax(output_word_fr, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr, output_word_en) / (seq_len_fr*self.batch_size)
        return l2_loss, loss, loss_2




    def forward(self, batch_input, lang='En', unlabeled=False):
        if unlabeled:

            loss = self.parallel_train(batch_input)

            loss_word = 0

            return loss, loss_word

        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        predicates_1D = batch_input['predicates_idx']
        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        word_id = get_torch_variable_from_np(batch_input['word_times'])
        word_id_emb = self.id_embedding(word_id)
        flag_emb = self.flag_embedding(flag_batch)

        if lang == "En":
            pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
        else:
            pretrain_emb = self.fr_pretrained_embedding(pretrain_batch).detach()

        seq_len = flag_emb.shape[1]
        SRL_output = self.SR_Labeler(pretrain_emb, flag_emb, predicates_1D, seq_len, para=False)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        SRL_input = SRL_input.detach()
        pred_recur = self.SR_Compressor(SRL_input, pretrain_emb, word_id_emb, seq_len, para=False)

        output_word = self.SR_Matcher(pred_recur, pretrain_emb, word_id_emb.detach(), seq_len, para=False)
        return SRL_output, output_word






