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

        if USE_CUDA:
            self.bilstm_hidden_state_word = (
            Variable(torch.randn(2 * 2, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda(),
            Variable(torch.randn(2 * 2, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True).cuda())
        else:
            self.bilstm_hidden_state_word = (
            Variable(torch.randn(2 * 2, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True),
            Variable(torch.randn(2 * 2, self.batch_size, self.bilstm_hidden_size),
                     requires_grad=True))


        self.bilstm_layer_word = nn.LSTM(input_size=300+self.target_vocab_size+2*self.flag_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=2,
                                    bidirectional=True,
                                    bias=True, batch_first=True)

    def forward(self, SRL_input, pretrained_emb, flag_emb, word_id_emb, predicates_1D, seq_len, para=False):
        SRL_input = SRL_input.view(self.batch_size, seq_len, -1)
        SRL_input = SRL_input.detach()
        input_emb_word = torch.cat((pretrained_emb, flag_emb), 2)
        if para == False:
            input_emb_word = self.dropout_word(input_emb_word)
        compress_input = torch.cat((input_emb_word, word_id_emb, SRL_input), 2)
        bilstm_output_word, (_, bilstm_final_state_word) = self.bilstm_layer_word(compress_input,
                                                                                  self.bilstm_hidden_state_word)
        bilstm_output_word = bilstm_output_word.contiguous()
        if para == False:
            bilstm_output_word = self.dropout_hidden(bilstm_output_word)
        pred_recur = bilstm_output_word[np.arange(0, self.batch_size), predicates_1D]
        return pred_recur


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
        self.match_word = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size+300+2*self.flag_emb_size, self.mlp_size),
                                        nn.ReLU(),
                                        nn.Linear(self.mlp_size, self.target_vocab_size))
    def forward(self, pred_recur, pretrained_emb, flag_emb, word_id_emb, seq_len, para=False):
        input_emb_word = torch.cat((pretrained_emb, flag_emb), 2)
        if para == False:
            input_emb_word = self.dropout_word(input_emb_word)
        pred_recur = pred_recur.view(self.batch_size, self.bilstm_hidden_size * 2)
        pred_recur = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, self.bilstm_hidden_size * 2)
        combine = torch.cat((pred_recur, input_emb_word, word_id_emb), 2)
        output_word = self.match_word(combine)
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
        # log(flag_batch_fr)
        word_id_fr = get_torch_variable_from_np(unlabeled_data_fr['word_times'])
        word_id_emb_fr = self.id_embedding(word_id_fr).detach()
        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()

        pretrain_emb_fr = self.fr_pretrained_embedding(pretrain_batch_fr).detach()
        pretrain_emb_fr_matrixed = self.word_matrix(pretrain_emb_fr)
        input_emb_fr = torch.cat((pretrain_emb_fr, flag_emb_fr), 2).detach()
        #input_emb_fr_matrixed = torch.cat((pretrain_emb_fr_matrixed, flag_emb_fr), 2).detach()
        seq_len_fr = input_emb_fr.shape[1]


        pretrain_batch = get_torch_variable_from_np(unlabeled_data_en['pretrain'])
        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        #log(flag_batch)
        word_id = get_torch_variable_from_np(unlabeled_data_en['word_times'])
        word_id_emb = self.id_embedding(word_id)
        flag_emb = self.flag_embedding(flag_batch)
        pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
        word_id_emb_en = word_id_emb.detach()
        pretrain_emb_en = pretrain_emb
        input_emb = torch.cat((pretrain_emb, flag_emb), 2).detach()
        #input_emb = self.word_dropout(input_emb)
        input_emb_en = input_emb.detach()
        seq_len = input_emb.shape[1]
        seq_len_en = seq_len
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state_p)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)
        arg_hidden = self.mlp_arg(hidden_input)
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.mlp_pred(pred_recur)
        SRL_output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                              num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        SRL_output = SRL_output.view(self.batch_size * seq_len, -1)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        compress_input = torch.cat((input_emb.detach(), word_id_emb.detach(), SRL_input.detach()), 2)
        bilstm_output_word, (_, bilstm_final_state_word) = self.bilstm_layer_word(compress_input,
                                                                                  self.bilstm_hidden_state_word_p)
        bilstm_output_word = bilstm_output_word.contiguous().detach()
        # hidden_input_word = bilstm_output_word.view(bilstm_output_word.shape[0] * bilstm_output_word.shape[1], -1)
        pred_recur = bilstm_output_word[np.arange(0, self.batch_size), predicates_1D]
        pred_recur = pred_recur.view(self.batch_size, self.bilstm_hidden_size * 2)
        pred_recur_1 = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, self.bilstm_hidden_size * 2)
        pred_recur_2 = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len_fr, self.bilstm_hidden_size * 2)
        pred_recur_en = pred_recur_1.detach()
        pred_recur_en_2 = pred_recur_2.detach()

        """
        En event vector, En word
        """
        combine = torch.cat((pred_recur_en.detach(), input_emb.detach(), word_id_emb.detach()), 2)
        output_word = self.match_word(combine)
        output_word_en = output_word.view(self.batch_size*seq_len, -1).detach()





        bilstm_output_fr, (_, bilstm_final_state) = self.bilstm_layer(input_emb_fr, self.bilstm_hidden_state_p)
        bilstm_output_fr = bilstm_output_fr.contiguous()
        hidden_input_fr = bilstm_output_fr.view(bilstm_output_fr.shape[0] * bilstm_output_fr.shape[1], -1)
        hidden_input_fr = hidden_input_fr.view(self.batch_size, seq_len_fr, -1)
        arg_hidden_fr = self.mlp_arg(hidden_input_fr)
        pred_recur_fr = hidden_input_fr[np.arange(0, self.batch_size), predicates_1D_fr]
        pred_hidden_fr = self.mlp_pred(pred_recur_fr)
        SRL_output_fr = bilinear(arg_hidden_fr, self.rel_W, pred_hidden_fr, self.mlp_size, seq_len_fr, 1, self.batch_size,
                              num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        SRL_output_fr = SRL_output_fr.view(self.batch_size * seq_len_fr, -1)

        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        compress_input_fr = torch.cat((input_emb_fr.detach(), word_id_emb_fr.detach(), SRL_input_fr), 2)
        bilstm_output_word_fr, (_, bilstm_final_state_word) = self.bilstm_layer_word(compress_input_fr,
                                                                                  self.bilstm_hidden_state_word_p)
        bilstm_output_word_fr = bilstm_output_word_fr.contiguous()
        pred_recur_fr = bilstm_output_word_fr[np.arange(0, self.batch_size), predicates_1D_fr]
        pred_recur_fr = pred_recur_fr.view(self.batch_size, self.bilstm_hidden_size * 2)

        #############################################
        """
        Fr event vector, En word
        """

        pred_recur_fr_1 = pred_recur_fr.unsqueeze(1).expand(self.batch_size, seq_len_en, self.bilstm_hidden_size * 2)
        combine = torch.cat((pred_recur_fr_1, input_emb_en.detach(), word_id_emb_en.detach()), 2)
        output_word_fr = self.match_word(combine)
        output_word_fr = output_word_fr.view(self.batch_size*seq_len_en, -1)

        unlabeled_loss_function = nn.KLDivLoss(size_average=False)
        output_word_en = F.softmax(output_word_en, dim=1).detach()
        output_word_fr = F.log_softmax(output_word_fr, dim=1)
        loss = unlabeled_loss_function(output_word_fr, output_word_en)/(seq_len_en*self.para_batch_size)

        #############################################3
        """
        En event vector, Fr word
        """
        combine = torch.cat((pred_recur_en_2.detach(), input_emb_fr.detach(), word_id_emb_fr.detach()), 2)
        output_word = self.match_word(combine)
        output_word_en_2 = output_word.view(self.batch_size * seq_len_fr, -1)

        """
        Fr event vector, Fr word
        """
        pred_recur_fr_2 = pred_recur_fr.unsqueeze(1).expand(self.batch_size, seq_len_fr, self.bilstm_hidden_size * 2)
        combine = torch.cat((pred_recur_fr_2, input_emb_fr.detach(), word_id_emb_fr.detach()), 2)
        output_word_fr_2 = self.match_word(combine)
        output_word_fr_2 = output_word_fr_2.view(self.batch_size * seq_len_fr, -1)

        unlabeled_loss_function = nn.KLDivLoss(size_average=False)
        output_word_en_2 = F.softmax(output_word_en_2, dim=1).detach()
        output_word_fr_2 = F.log_softmax(output_word_fr_2, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr_2, output_word_en_2) / (seq_len_fr*self.para_batch_size)
        return loss, loss_2


    def word_train(self, batch_input):
        unlabeled_data_en, unlabeled_data_fr = batch_input

        pretrain_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['pretrain'])
        predicates_1D_fr = unlabeled_data_fr['predicates_idx']
        flag_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['flag'])
        # log(flag_batch_fr)
        word_id_fr = get_torch_variable_from_np(unlabeled_data_fr['word_times'])
        word_id_emb_fr = self.id_embedding(word_id_fr).detach()
        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()
        pretrain_emb_fr = self.fr_pretrained_embedding(pretrain_batch_fr).detach()
        pretrain_emb_fr = self.word_matrix(pretrain_emb_fr)
        input_emb_fr = torch.cat((pretrain_emb_fr, flag_emb_fr), 2)
        seq_len_fr = input_emb_fr.shape[1]


        word_batch = get_torch_variable_from_np(unlabeled_data_en['word'])
        pretrain_batch = get_torch_variable_from_np(unlabeled_data_en['pretrain'])
        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        #log(flag_batch)
        word_id = get_torch_variable_from_np(unlabeled_data_en['word_times'])
        word_id_emb = self.id_embedding(word_id)
        flag_emb = self.flag_embedding(flag_batch)
        pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
        word_id_emb_en = word_id_emb.detach()
        pretrain_emb_en = pretrain_emb
        input_emb = torch.cat((pretrain_emb, flag_emb), 2)
        #input_emb = self.word_dropout(input_emb)
        input_emb_en = input_emb
        seq_len = input_emb.shape[1]
        seq_len_en = seq_len
        bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state_p)
        bilstm_output = bilstm_output.contiguous()
        hidden_input = bilstm_output.view(bilstm_output.shape[0] * bilstm_output.shape[1], -1)
        hidden_input = hidden_input.view(self.batch_size, seq_len, -1)
        arg_hidden = self.mlp_arg(hidden_input)
        pred_recur = hidden_input[np.arange(0, self.batch_size), predicates_1D]
        pred_hidden = self.mlp_pred(pred_recur)
        SRL_output = bilinear(arg_hidden, self.rel_W, pred_hidden, self.mlp_size, seq_len, 1, self.batch_size,
                              num_outputs=self.target_vocab_size, bias_x=True, bias_y=True)
        SRL_output = SRL_output.view(self.batch_size * seq_len, -1)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        compress_input = torch.cat((input_emb, word_id_emb, SRL_input), 2)
        bilstm_output_word, (_, bilstm_final_state_word) = self.bilstm_layer_word(compress_input,
                                                                                  self.bilstm_hidden_state_word_p)
        bilstm_output_word = bilstm_output_word.contiguous()
        # hidden_input_word = bilstm_output_word.view(bilstm_output_word.shape[0] * bilstm_output_word.shape[1], -1)
        pred_recur = bilstm_output_word[np.arange(0, self.batch_size), predicates_1D]
        pred_recur = pred_recur.view(self.batch_size, self.bilstm_hidden_size * 2)
        pred_recur_1 = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, self.bilstm_hidden_size * 2)
        pred_recur_2 = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len_fr, self.bilstm_hidden_size * 2)
        pred_recur_en = pred_recur_1
        pred_recur_en_2 = pred_recur_2

        combine = torch.cat((pred_recur_en, input_emb, word_id_emb), 2)
        output_word = self.match_word(combine)
        output_word_en = output_word.view(self.batch_size, seq_len, -1)
        output_word_en = F.softmax(output_word_en, 2)
        max_role_en = torch.max(output_word_en, 1)[0].detach()

        combine_fr = torch.cat((pred_recur_en_2.detach(), input_emb_fr, word_id_emb_fr.detach()), 2)
        output_word_fr = self.match_word(combine_fr)
        output_word_fr = output_word_fr.view(self.batch_size, seq_len_fr, -1)
        output_word_fr = F.softmax(output_word_fr, 2)
        max_role_fr = torch.max(output_word_fr, 1)[0]
        loss = nn.MSELoss()
        word_loss = loss(max_role_fr, max_role_en)
        return word_loss


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
        SRL_input = SRL_input
        pred_recur = self.SR_Compressor(SRL_input, pretrain_emb,
                                        flag_emb.detach(), word_id_emb, predicates_1D, seq_len, para=False)

        output_word = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), word_id_emb.detach(), seq_len, para=False)
        return SRL_output, output_word






