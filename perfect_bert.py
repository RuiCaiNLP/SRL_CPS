from __future__ import print_function
import numpy as np
import torch
from torch import nn
import sys
# from torch.autograd import Variable
import torch.nn.functional as F
from transformers import *
from pytorch_revgrad import RevGrad


# from utils import USE_CUDA
from utils import get_torch_variable_from_np, get_data
from utils import bilinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise
    return ins


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

        self.bilstm_hidden_state = (
            torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size).to(device),
            torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size).to(device))

        self.bilstm_layer = nn.LSTM(input_size=300 + 1 * self.flag_emb_size,
                                    hidden_size=self.bilstm_hidden_size, num_layers=self.bilstm_num_layers,
                                    bidirectional=True,
                                    bias=True, batch_first=True)

        self.bilstm_bert = nn.LSTM(input_size=768 + self.flag_emb_size,
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

    def forward(self, pretrained_emb, flag_emb, predicates_1D, seq_len, use_bert=False, para=False):
        self.bilstm_hidden_state = (
            torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size).to(device),
            torch.zeros(2 * self.bilstm_num_layers, self.batch_size, self.bilstm_hidden_size).to(device))

        input_emb = torch.cat((pretrained_emb, flag_emb), 2)
        if para == False:
            input_emb = self.dropout_word(input_emb)
        if not use_bert:
            bilstm_output, (_, bilstm_final_state) = self.bilstm_layer(input_emb, self.bilstm_hidden_state)
        else:
            bilstm_output, (_, bilstm_final_state) = self.bilstm_bert(input_emb, self.bilstm_hidden_state)

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
        #self.dropout_word = nn.Dropout(p=0.0)
        #self.dropout_hidden = nn.Dropout(p=0.0)
        #self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']

        self.bilstm_hidden_state_probs = (
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size).to(device),
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size).to(device))

        self.bilstm_layer_probs = nn.LSTM(input_size=self.target_vocab_size,
                                         hidden_size=self.target_vocab_size, num_layers=2,
                                         bidirectional=True,
                                         bias=True, batch_first=True)
        self.combine_wordprobs = nn.Sequential(nn.Linear(768 + 2 * self.target_vocab_size, 20*(self.target_vocab_size-1)),
                                               nn.LeakyReLU(0.1),
                                               nn.Linear(20 * (self.target_vocab_size - 1),
                                                         20 * (self.target_vocab_size - 1)),
                                               nn.LeakyReLU(0.1))

    def forward(self, SRL_input_probs, word_emb, flag_emb, word_id_emb, predicates_1D, seq_len, use_bert=False, para=False):
        self.bilstm_hidden_state_probs = (
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size).to(device),
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size).to(device))

        SRL_input_probs = SRL_input_probs.view(self.batch_size, seq_len, -1)
        bilstm_output_probs, (_, bilstm_final_state_probs) = self.bilstm_layer_probs(SRL_input_probs,
                                                             self.bilstm_hidden_state_probs)
        # BT 2R
        bilstm_output = bilstm_output_probs.contiguous()
        word_probs = torch.cat((bilstm_output, word_emb), 2)
        converted_wordprobs = self.combine_wordprobs(word_probs)
        pred_recur = torch.max(converted_wordprobs, dim=1)[0]
        return pred_recur


class SR_Matcher(nn.Module):
    def __init__(self, model_params):
        super(SR_Matcher, self).__init__()
        #self.dropout_word_1 = nn.Dropout(p=0.0)
        #self.dropout_word_2 = nn.Dropout(p=0.0)
        self.mlp_size = 300
        #self.dropout_mlp = model_params['dropout_mlp']
        self.batch_size = model_params['batch_size']

        self.target_vocab_size = model_params['target_vocab_size']
        self.use_flag_embedding = model_params['use_flag_embedding']
        self.flag_emb_size = model_params['flag_embedding_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.bilstm_num_layers = model_params['bilstm_num_layers']
        self.bilstm_hidden_size = model_params['bilstm_hidden_size']
        self.compress_word = nn.Sequential(nn.Linear(300 + 0 * self.flag_emb_size, 20), nn.LeakyReLU(0.1))
        self.compress_bert = nn.Sequential(nn.Linear(768 + 0 * self.flag_emb_size, 20), nn.LeakyReLU(0.1))
        self.hidden2compression =  nn.Sequential(nn.Linear((self.target_vocab_size-1)*20, (self.target_vocab_size-1)*20),
                                                 nn.LeakyReLU(0.1))
        self.scorer = nn.Sequential(nn.Linear(40, 20),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(20, 1))
        self.match_word = nn.Sequential(
            nn.Linear(2 * self.bilstm_hidden_size + 300 + 2 * self.flag_emb_size, self.mlp_size),
            nn.Tanh(),
            nn.Linear(self.mlp_size, self.target_vocab_size))

    def forward(self, pred_recur, pretrained_emb, flag_emb, word_id_emb, seq_len, use_bert=False, copy = False, para=False):
        """
        pred_recur = pred_recur.view(self.batch_size, self.bilstm_hidden_size * 2)
        pred_recur = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, self.bilstm_hidden_size * 2)
        combine = torch.cat((pred_recur, pretrained_emb, flag_emb, word_id_emb), 2)
        output_word = self.match_word(combine)
        output_word = output_word.view(self.batch_size * seq_len, -1)
        """

        #forward_hidden = pred_recur[:, :(self.target_vocab_size-1) * 10].view(self.batch_size, (self.target_vocab_size-1),
        #                                                                  10)
        #backward_hidden = pred_recur[:, (self.target_vocab_size-1) * 10:].view(self.batch_size, (self.target_vocab_size-1),
        #                                                                   10)
        #role_hidden = torch.cat((forward_hidden, backward_hidden), 2)
        role_hidden = pred_recur.view(self.batch_size, self.target_vocab_size-1,20)
        role_hidden = role_hidden.unsqueeze(1).expand(self.batch_size, seq_len, self.target_vocab_size-1, 20)
        #if copy:
        #    pretrained_emb = self.dropout_word_1(pretrained_emb)
        #if use_bert:
        #    combine = self.compress_bert(pretrained_emb)
        #else:
        combine = self.compress_bert(pretrained_emb)
        combine = combine.unsqueeze(2).expand(self.batch_size, seq_len, self.target_vocab_size-1, 20)
        scores = self.scorer(torch.cat((role_hidden, combine), 3)).view(self.batch_size * seq_len,
                                                                        (self.target_vocab_size - 1))

        return scores

class Discriminator(nn.Module):
    def __init__(self, model_params):
        super(Discriminator, self).__init__()
        """                        
        self.emb_dim = 256
        self.dis_hid_dim = 200
        self.dis_layers = 1
        self.dis_input_dropout = 0.2
        self.dis_dropout = 0.2
        layers = []#[RevGrad()]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 2 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        #layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        """
        self.Classifier = nn.Sequential(RevGrad(),
                                        nn.Linear(256, 2))
    def forward(self, x):
        #assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.Classifier(x).view(-1)


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
        self.word_matrix.weight.data.copy_(
            torch.from_numpy(np.eye(self.pretrain_emb_size, self.pretrain_emb_size, dtype="float32")))

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

        self.Discriminator = Discriminator(model_params)
        self.real = 1#np.random.uniform(0.99, 1.0)  # 1
        self.fake = 0#np.random.uniform(0.0, 0.01)  # 0


        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()

        self.bert_FeatureExtractor = nn.Sequential(nn.Linear(768, 256),
                                        #nn.LeakyReLU(0.2),
                                        #nn.Linear(512, 256),
                                        nn.Tanh())


        #self.Fr_LinearTrans.weight.data.copy_(
        #    torch.from_numpy(np.eye(768, 768, dtype="float32")))
        #self.En_LinearTrans.weight.data.copy_(
        #    torch.from_numpy(np.eye(768, 768, dtype="float32")))



    def copy_loss(self, output_SRL, pretrain_emb, flag_emb, seq_len):
        SRL_input = output_SRL.view(self.batch_size, seq_len, -1)
        SRL_input = F.softmax(SRL_input, 2).detach()
        pred_recur = self.SR_Compressor(SRL_input, pretrain_emb,
                                        flag_emb.detach(), None, None, seq_len, para=False, use_bert=True)

        output_word = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), None, seq_len, copy = True,
                                      para=False, use_bert=True)

        score4Null = torch.zeros_like(output_word[:, 1:2])
        output_word = torch.cat((output_word[:, 0:1], score4Null, output_word[:, 1:]), 1)

        teacher = SRL_input.view(self.batch_size * seq_len, -1).detach()#F.softmax(SRL_input.view(self.batch_size * seq_len, -1), dim=1).detach()
        student = F.log_softmax(output_word.view(self.batch_size * seq_len, -1), dim=1)
        unlabeled_loss_function = nn.KLDivLoss(reduction='none')
        loss = unlabeled_loss_function(student, teacher)
        loss = loss.sum() / (self.batch_size * seq_len)
        return loss


    def learn_loss(self, output_SRL, pretrain_emb, flag_emb, seq_len, mask_copy, mask_unk):
        SRL_input = output_SRL.view(self.batch_size, seq_len, -1)
        output = SRL_input.view(self.batch_size*seq_len, -1)
        SRL_input = F.softmax(SRL_input, 2).detach()

        pred = torch.max(SRL_input, dim=2)[1]
        for i in range(self.batch_size):
            for j in range(seq_len):
                if pred[i][j] > 1:
                    mask_copy[i][j] = 1

        mask_copy = get_torch_variable_from_np(mask_copy)
        mask_final = mask_copy.view(self.batch_size * seq_len) * mask_unk.view(self.batch_size * seq_len)
        pred_recur = self.SR_Compressor(SRL_input, pretrain_emb,
                                        flag_emb.detach(), None, None, seq_len, para=False, use_bert=True)

        output_word = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), None, seq_len, copy = True,
                                      para=False, use_bert=True)

        score4Null = torch.zeros_like(output_word[:, 1:2])
        output_word = torch.cat((output_word[:, 0:1], score4Null, output_word[:, 1:]), 1)

        criterion = nn.CrossEntropyLoss(reduction='none')
        _, prediction_batch_variable = torch.max(output, 1)

        loss_word = criterion(output_word, prediction_batch_variable)*mask_final

        loss_word = loss_word.sum()/(self.batch_size*seq_len)


        return loss_word


    def filter_word(self, SRL_pred, SRL_out_en_en, SRL_out_en_fr, seq_len, seq_len_fr):
        SRL_pred = SRL_pred.view(self.batch_size, seq_len, self.target_vocab_size)
        SRL_out_en_en = SRL_out_en_en.view(self.batch_size, seq_len, self.target_vocab_size)
        SRL_out_en_fr = SRL_out_en_fr.view(self.batch_size, seq_len_fr, self.target_vocab_size)
        result_en = torch.max(SRL_pred, dim=2)[1].view(self.batch_size, seq_len)
        result_en_en = torch.max(SRL_out_en_en, dim=2)[1].view(self.batch_size, seq_len)
        result_en_fr = torch.max(SRL_out_en_fr, dim=2)[1].view(self.batch_size, seq_len_fr)

        for i in range(self.batch_size):
            for j in range(seq_len):
                if result_en[i][j] != result_en_en[i][j]:
                    result_en_en[i][j] = 1
        mask_en = np.zeros((self.batch_size, seq_len))
        mask_fr = np.zeros((self.batch_size, seq_len_fr))
        roles_en_en = np.zeros((self.batch_size, self.target_vocab_size))-1
        roles_en_fr = np.zeros((self.batch_size, self.target_vocab_size))-1

        # -1: no this role
        # >=0: index of arguments
        # -2: duplicate
        for i in range(self.batch_size):
            for j in range(seq_len):
                this_result = result_en_en[i][j]
                if this_result > 1:
                    if roles_en_en[i][this_result]!= -1:
                        roles_en_en[i][this_result] = -2
                    else:
                        roles_en_en[i][this_result] = j

        for i in range(self.batch_size):
            for j in range(seq_len_fr):
                this_result = result_en_fr[i][j]
                if this_result > 1:
                    if roles_en_fr[i][this_result]!= -1:
                        roles_en_fr[i][this_result] = -2
                    else:
                        roles_en_fr[i][this_result] = j

        for i in range(self.batch_size):
            for j in range(self.target_vocab_size):
                if j <=1:
                    continue
                if roles_en_en[i][j]>=0 and roles_en_fr[i][j]>=0:
                    mask_en[i][int(roles_en_en[i][j])] = 1
                    mask_fr[i][int(roles_en_fr[i][j])] = 1

        return mask_en, mask_fr

    def filter_word_fr(self, SRL_out_fr_en, SRL_out_fr_fr, seq_len, seq_len_fr):
        SRL_out_fr_en = SRL_out_fr_en.view(self.batch_size, seq_len, self.target_vocab_size)
        SRL_out_fr_fr = SRL_out_fr_fr.view(self.batch_size, seq_len_fr, self.target_vocab_size)

        result_fr_en = torch.max(SRL_out_fr_en, dim=2)[1].view(self.batch_size, seq_len)
        result_fr_fr = torch.max(SRL_out_fr_fr, dim=2)[1].view(self.batch_size, seq_len_fr)


        mask_en = np.zeros((self.batch_size, seq_len))
        mask_fr = np.zeros((self.batch_size, seq_len_fr))
        roles_fr_en = np.zeros((self.batch_size, self.target_vocab_size))-1
        roles_fr_fr = np.zeros((self.batch_size, self.target_vocab_size))-1

        # -1: no this role
        # >=0: index of arguments
        # -2: duplicate
        for i in range(self.batch_size):
            for j in range(seq_len):
                this_result = result_fr_en[i][j]
                if this_result > 1:
                    if roles_fr_en[i][this_result]!= -1:
                        roles_fr_en[i][this_result] = -2
                    else:
                        roles_fr_en[i][this_result] = j

        for i in range(self.batch_size):
            for j in range(seq_len_fr):
                this_result = result_fr_fr[i][j]
                if this_result > 1:
                    if roles_fr_fr[i][this_result]!= -1:
                        roles_fr_fr[i][this_result] = -2
                    else:
                        roles_fr_fr[i][this_result] = j

        for i in range(self.batch_size):
            for j in range(self.target_vocab_size):
                if j <=1:
                    continue
                if roles_fr_en[i][j]>=0 and roles_fr_fr[i][j]>=0:
                    mask_en[i][int(roles_fr_en[i][j])] = 1
                    mask_fr[i][int(roles_fr_fr[i][j])] = 1

        return mask_en, mask_fr

    def parallel_train_(self, batch_input, use_bert, isTrain=True):
        unlabeled_data_en, unlabeled_data_fr = batch_input

        predicates_1D_fr = unlabeled_data_fr['predicates_idx']
        flag_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['flag'])

        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()
        actual_lens_fr = unlabeled_data_fr['seq_len']

        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        actual_lens_en = unlabeled_data_en['seq_len']
        flag_emb = self.flag_embedding(flag_batch).detach()
        seq_len = flag_emb.shape[1]
        seq_len_en = seq_len

        pretrain_batch_en = get_torch_variable_from_np(unlabeled_data_en['pretrain'])
        pretrain_emb_en = self.pretrained_embedding(pretrain_batch_en).detach()

        pretrain_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['pretrain'])
        pretrain_emb_fr = self.pretrained_embedding(pretrain_batch_fr).detach()

        bert_input_ids_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_input_ids'])
        bert_input_mask_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_input_mask'])
        bert_out_positions_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_out_positions'])

        bert_emb_fr = self.model(bert_input_ids_fr, attention_mask=bert_input_mask_fr)
        bert_emb_fr = bert_emb_fr[0]
        bert_emb_fr = bert_emb_fr[:, 1:-1, :].contiguous().detach()
        bert_emb_fr = bert_emb_fr[torch.arange(bert_emb_fr.size(0)).unsqueeze(-1), bert_out_positions_fr].detach()

        for i in range(len(bert_emb_fr)):
            if i >= len(actual_lens_fr):
                print("error")
                break
            for j in range(len(bert_emb_fr[i])):
                if j >= actual_lens_fr[i]:
                    bert_emb_fr[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
        bert_emb_fr_noise = gaussian(bert_emb_fr, isTrain, 0, 0.1).detach()
        bert_emb_fr = bert_emb_fr.detach()

        bert_input_ids_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_ids'])
        bert_input_mask_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_mask'])
        bert_out_positions_en = get_torch_variable_from_np(unlabeled_data_en['bert_out_positions'])

        bert_emb_en = self.model(bert_input_ids_en, attention_mask=bert_input_mask_en)
        bert_emb_en = bert_emb_en[0]
        bert_emb_en = bert_emb_en[:, 1:-1, :].contiguous().detach()
        bert_emb_en = bert_emb_en[torch.arange(bert_emb_en.size(0)).unsqueeze(-1), bert_out_positions_en].detach()

        for i in range(len(bert_emb_en)):
            if i >= len(actual_lens_en):
                print("error")
                break
            for j in range(len(bert_emb_en[i])):
                if j >= actual_lens_en[i]:
                    bert_emb_en[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
        bert_emb_en_noise = gaussian(bert_emb_en, isTrain, 0, 0.1).detach()
        bert_emb_en = bert_emb_en.detach()

        mask_para_en = get_torch_variable_from_np(unlabeled_data_en['mask_para'])
        mask_copy_en = unlabeled_data_en['mask_para']
        mask_unk_en = get_torch_variable_from_np(unlabeled_data_en['mask_unk'])
        mask_final_en = mask_para_en.view(self.batch_size * seq_len) * mask_unk_en.view(self.batch_size * seq_len)

        seq_len_fr = flag_emb_fr.shape[1]
        mask_para_fr = get_torch_variable_from_np(unlabeled_data_fr['mask_para'])
        mask_copy_fr = unlabeled_data_fr['mask_para']
        mask_unk_fr = get_torch_variable_from_np(unlabeled_data_fr['mask_unk'])
        mask_final_fr = mask_para_fr.view(self.batch_size * seq_len_fr) * mask_unk_fr.view(self.batch_size * seq_len_fr)

        seq_len = flag_emb.shape[1]
        SRL_output = self.SR_Labeler(bert_emb_en, flag_emb.detach(), predicates_1D, seq_len, para=True, use_bert=True)

        #CopyLoss_en_noise = self.copy_loss(SRL_output, bert_emb_en_noise, flag_emb.detach(), seq_len)
        CopyLoss_en = self.learn_loss(SRL_output, pretrain_emb_en, flag_emb.detach(), seq_len, mask_copy_en, mask_unk_en)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        SRL_input = F.softmax(SRL_input, 2)
        pred_recur = self.SR_Compressor(SRL_input.detach(), pretrain_emb_en,
                                        flag_emb.detach(), None, predicates_1D, seq_len, para=True, use_bert=True)

        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(bert_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True,
                                        use_bert=True)

        #CopyLoss_fr_noise = self.copy_loss(SRL_output_fr, bert_emb_fr_noise, flag_emb_fr.detach(), seq_len_fr)
        CopyLoss_fr = self.learn_loss(SRL_output_fr, pretrain_emb_fr, flag_emb_fr.detach(), seq_len_fr,mask_copy_fr, mask_unk_fr)


        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        SRL_input_fr = F.softmax(SRL_input_fr, 2)
        pred_recur_fr = self.SR_Compressor(SRL_input_fr, pretrain_emb_fr,
                                           flag_emb_fr.detach(), None, predicates_1D_fr, seq_len_fr, para=True,
                                           use_bert=True)


        """
        En event vector, En word
        """
        output_word_en_en = self.SR_Matcher(pred_recur.detach(), pretrain_emb_en, flag_emb.detach(), None, seq_len,
                                            para=True, use_bert=True).detach()
        score4Null = torch.zeros_like(output_word_en_en[:, 1:2])
        output_word_en_en = torch.cat((output_word_en_en[:, 0:1], score4Null, output_word_en_en[:, 1:]), 1)

        #############################################
        """
        Fr event vector, En word
        """
        output_word_fr_en = self.SR_Matcher(pred_recur_fr, pretrain_emb_en, flag_emb.detach(), None, seq_len,
                                            para=True, use_bert=True)
        score4Null = torch.zeros_like(output_word_fr_en[:, 1:2])
        output_word_fr_en = torch.cat((output_word_fr_en[:, 0:1], score4Null, output_word_fr_en[:, 1:]), 1)

        #############################################3
        """
        En event vector, Fr word
        """
        output_word_en_fr = self.SR_Matcher(pred_recur.detach(), pretrain_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                            para=True, use_bert=True).detach()
        score4Null = torch.zeros_like(output_word_en_fr[:, 1:2])
        output_word_en_fr = torch.cat((output_word_en_fr[:, 0:1], score4Null, output_word_en_fr[:, 1:]), 1)

        """
        Fr event vector, Fr word
         """

        output_word_fr_fr = self.SR_Matcher(pred_recur_fr, pretrain_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                            para=True, use_bert=True)
        score4Null = torch.zeros_like(output_word_fr_fr[:, 1:2])
        output_word_fr_fr = torch.cat((output_word_fr_fr[:, 0:1], score4Null, output_word_fr_fr[:, 1:]), 1)

        mask_en_en, mask_en_fr = self.filter_word(SRL_input, output_word_en_en,output_word_en_fr, seq_len, seq_len_fr)
        mask_en_en = get_torch_variable_from_np(mask_en_en)
        mask_en_fr = get_torch_variable_from_np(mask_en_fr)

        mask_fr_en, mask_fr_fr = self.filter_word_fr(output_word_fr_en, output_word_fr_fr, seq_len, seq_len_fr)
        mask_fr_en = get_torch_variable_from_np(mask_fr_en)
        mask_fr_fr = get_torch_variable_from_np(mask_fr_fr)

        mask_en_word = mask_en_en + mask_fr_en - mask_en_en*mask_fr_en
        mask_fr_word = mask_en_fr + mask_fr_fr - mask_en_fr*mask_fr_fr


        unlabeled_loss_function = nn.KLDivLoss(reduction='none')

        # output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        # output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        # loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en)
        #output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en).sum(dim=1)*mask_final_en*mask_en_word.view(-1)
        #loss = loss.sum() / mask_en_word.sum() #(self.batch_size * seq_len_en)
        #if mask_en_word.sum().cpu().numpy() > 1:
        loss = loss.sum() / (self.batch_size * seq_len)
        #else:
        #    loss = loss.sum()

        # output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        output_word_fr_fr = F.log_softmax(output_word_fr_fr, dim=1)
        #output_word_fr_fr = F.log_softmax(SRL_output_fr, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr_fr, output_word_en_fr).sum(dim=1)*mask_final_fr*mask_fr_word.view(-1)
        #if mask_fr_word.sum().cpu().numpy() > 1:
        loss_2 = loss_2.sum() / (self.batch_size * seq_len_fr)
        #else:
        #    loss_2 = loss_2.sum()

        #print(mask_en_word.sum())
        #print(mask_fr_word.sum())


        return loss, loss_2, CopyLoss_en, CopyLoss_fr

    def word_trans(self, batch_input, use_bert, isTrain=True):
        unlabeled_data_en, unlabeled_data_fr = batch_input

        predicates_1D_fr = unlabeled_data_fr['predicates_idx']
        flag_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['flag'])

        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()
        actual_lens_fr = unlabeled_data_fr['seq_len']

        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        actual_lens_en = unlabeled_data_en['seq_len']
        flag_emb = self.flag_embedding(flag_batch).detach()
        seq_len = flag_emb.shape[1]
        seq_len_en = seq_len

        if use_bert:
            bert_input_ids_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_input_ids'])
            bert_input_mask_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_input_mask'])
            bert_out_positions_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_out_positions'])

            bert_emb_fr = self.model(bert_input_ids_fr, attention_mask=bert_input_mask_fr)
            bert_emb_fr = bert_emb_fr[0]
            bert_emb_fr = bert_emb_fr[:, 1:-1, :].contiguous().detach()
            bert_emb_fr = bert_emb_fr[torch.arange(bert_emb_fr.size(0)).unsqueeze(-1), bert_out_positions_fr].detach()

            for i in range(len(bert_emb_fr)):
                if i >= len(actual_lens_fr):
                    print("error")
                    break
                for j in range(len(bert_emb_fr[i])):
                    if j >= actual_lens_fr[i]:
                        bert_emb_fr[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
            bert_emb_fr_noise = gaussian(bert_emb_fr, isTrain, 0, 0.1).detach()
            bert_emb_fr = bert_emb_fr.detach()

            bert_input_ids_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_ids'])
            bert_input_mask_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_mask'])
            bert_out_positions_en = get_torch_variable_from_np(unlabeled_data_en['bert_out_positions'])

            bert_emb_en = self.model(bert_input_ids_en, attention_mask=bert_input_mask_en)
            bert_emb_en = bert_emb_en[0]
            bert_emb_en = bert_emb_en[:, 1:-1, :].contiguous().detach()
            bert_emb_en = bert_emb_en[torch.arange(bert_emb_en.size(0)).unsqueeze(-1), bert_out_positions_en].detach()

            for i in range(len(bert_emb_en)):
                if i >= len(actual_lens_en):
                    print("error")
                    break
                for j in range(len(bert_emb_en[i])):
                    if j >= actual_lens_en[i]:
                        bert_emb_en[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
            bert_emb_en_noise = gaussian(bert_emb_en, isTrain, 0, 0.1).detach()
            bert_emb_en = bert_emb_en.detach()

            pred_bert_fr = bert_emb_fr[np.arange(0, self.batch_size), predicates_1D_fr]
            pred_bert_en = bert_emb_en[np.arange(0, self.batch_size), predicates_1D]
            #transed_bert_fr = self.Fr2En_Trans(pred_bert_fr)
            En_Extracted = self.bert_FeatureExtractor(pred_bert_en)
            Fr_Extracted = self.bert_FeatureExtractor(pred_bert_fr)
            #loss = nn.MSELoss()
            #l2loss = loss(Fr_Extracted, En_Extracted)
            #return l2loss

            x_D_real = En_Extracted.view(-1, 256)#self.bert_NonlinearTrans(pred_bert_en.detach().view(-1, 768))
            x_D_fake = Fr_Extracted.view(-1, 256)
            #x_D_real = self.En_LinearTrans(pred_bert_en.detach()).view(-1, 768)
            #x_D_fake = self.Fr_LinearTrans(pred_bert_fr.detach()).view(-1, 768)
            en_preds = self.Discriminator(x_D_real).view(self.batch_size, 2)
            real_labels = torch.empty((30,1), dtype=torch.long).fill_(1).view(-1)
            #D_loss_real = F.binary_cross_entropy(en_preds, real_labels)
            fr_preds = self.Discriminator(x_D_fake).view(self.batch_size, 2)
            fake_labels = torch.empty((30,1), dtype=torch.long).fill_(0).view(-1)
            #D_loss_fake = F.binary_cross_entropy(fr_preds, fake_labels)
            #D_loss = 0.5 * (D_loss_real + D_loss_fake)
            preds = torch.cat((en_preds, fr_preds), 0)
            labels = torch.cat((real_labels, fake_labels)).to(device)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(preds, labels)
            return loss


    def parallel_train_(self, batch_input, use_bert, isTrain=True):
        unlabeled_data_en, unlabeled_data_fr = batch_input

        predicates_1D_fr = unlabeled_data_fr['predicates_idx']
        flag_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['flag'])

        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()
        actual_lens_fr = unlabeled_data_fr['seq_len']

        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        actual_lens_en = unlabeled_data_en['seq_len']
        flag_emb = self.flag_embedding(flag_batch).detach()
        seq_len = flag_emb.shape[1]
        seq_len_en = seq_len

        if use_bert:
            bert_input_ids_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_input_ids'])
            bert_input_mask_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_input_mask'])
            bert_out_positions_fr = get_torch_variable_from_np(unlabeled_data_fr['bert_out_positions'])

            bert_emb_fr = self.model(bert_input_ids_fr, attention_mask=bert_input_mask_fr)
            bert_emb_fr = bert_emb_fr[0]
            bert_emb_fr = bert_emb_fr[:, 1:-1, :].contiguous().detach()
            bert_emb_fr = bert_emb_fr[torch.arange(bert_emb_fr.size(0)).unsqueeze(-1), bert_out_positions_fr].detach()

            for i in range(len(bert_emb_fr)):
                if i >= len(actual_lens_fr):
                    print("error")
                    break
                for j in range(len(bert_emb_fr[i])):
                    if j >= actual_lens_fr[i]:
                        bert_emb_fr[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
            bert_emb_fr_noise = gaussian(bert_emb_fr, isTrain, 0, 0.1).detach()
            bert_emb_fr = bert_emb_fr.detach()

            bert_input_ids_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_ids'])
            bert_input_mask_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_mask'])
            bert_out_positions_en = get_torch_variable_from_np(unlabeled_data_en['bert_out_positions'])

            bert_emb_en = self.model(bert_input_ids_en, attention_mask=bert_input_mask_en)
            bert_emb_en = bert_emb_en[0]
            bert_emb_en = bert_emb_en[:, 1:-1, :].contiguous().detach()
            bert_emb_en = bert_emb_en[torch.arange(bert_emb_en.size(0)).unsqueeze(-1), bert_out_positions_en].detach()

            for i in range(len(bert_emb_en)):
                if i >= len(actual_lens_en):
                    print("error")
                    break
                for j in range(len(bert_emb_en[i])):
                    if j >= actual_lens_en[i]:
                        bert_emb_en[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
            bert_emb_en_noise = gaussian(bert_emb_en, isTrain, 0, 0.1).detach()
            bert_emb_en = bert_emb_en.detach()

        seq_len = flag_emb.shape[1]
        SRL_output = self.SR_Labeler(bert_emb_en, flag_emb.detach(), predicates_1D, seq_len, para=True, use_bert=True)

        CopyLoss_en_noise = self.copy_loss(SRL_output, bert_emb_en_noise, flag_emb.detach(), seq_len)
        CopyLoss_en = self.copy_loss(SRL_output, bert_emb_en, flag_emb.detach(), seq_len)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        SRL_input = F.softmax(SRL_input, 2)
        pred_recur = self.SR_Compressor(SRL_input.detach(), bert_emb_en,
                                        flag_emb.detach(), None, predicates_1D, seq_len, para=True, use_bert=True)

        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(bert_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True,
                                        use_bert=True)

        CopyLoss_fr_noise = self.copy_loss(SRL_output_fr, bert_emb_fr_noise, flag_emb_fr.detach(), seq_len_fr)
        CopyLoss_fr = self.copy_loss(SRL_output_fr, bert_emb_fr, flag_emb_fr.detach(), seq_len_fr)


        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        SRL_input_fr = F.softmax(SRL_input_fr, 2)
        pred_recur_fr = self.SR_Compressor(SRL_input_fr, bert_emb_fr,
                                           flag_emb_fr.detach(), None, predicates_1D_fr, seq_len_fr, para=True,
                                           use_bert=True)


        """
        En event vector, En word
        """
        output_word_en_en = self.SR_Matcher(pred_recur.detach(), bert_emb_en, flag_emb.detach(), None, seq_len,
                                            para=True, use_bert=True).detach()
        score4Null = torch.zeros_like(output_word_en_en[:, 1:2])
        output_word_en_en = torch.cat((output_word_en_en[:, 0:1], score4Null, output_word_en_en[:, 1:]), 1)

        #############################################
        """
        Fr event vector, En word
        """
        output_word_fr_en = self.SR_Matcher(pred_recur_fr, bert_emb_en, flag_emb.detach(), None, seq_len,
                                            para=True, use_bert=True)
        score4Null = torch.zeros_like(output_word_fr_en[:, 1:2])
        output_word_fr_en = torch.cat((output_word_fr_en[:, 0:1], score4Null, output_word_fr_en[:, 1:]), 1)

        #############################################3
        """
        En event vector, Fr word
        """
        output_word_en_fr = self.SR_Matcher(pred_recur.detach(), bert_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                            para=True, use_bert=True).detach()
        score4Null = torch.zeros_like(output_word_en_fr[:, 1:2])
        output_word_en_fr = torch.cat((output_word_en_fr[:, 0:1], score4Null, output_word_en_fr[:, 1:]), 1)

        """
        Fr event vector, Fr word
        """
        output_word_fr_fr = self.SR_Matcher(pred_recur_fr, bert_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                            para=True, use_bert=True)
        score4Null = torch.zeros_like(output_word_fr_fr[:, 1:2])
        output_word_fr_fr = torch.cat((output_word_fr_fr[:, 0:1], score4Null, output_word_fr_fr[:, 1:]), 1)


        """
        mask_en_en, mask_en_fr = self.filter_word(SRL_input, output_word_en_en,output_word_en_fr, seq_len, seq_len_fr)
        mask_en_en = get_torch_variable_from_np(mask_en_en)
        mask_en_fr = get_torch_variable_from_np(mask_en_fr)

        mask_fr_en, mask_fr_fr = self.filter_word_fr(output_word_fr_en, output_word_fr_fr, seq_len, seq_len_fr)
        mask_fr_en = get_torch_variable_from_np(mask_fr_en)
        mask_fr_fr = get_torch_variable_from_np(mask_fr_fr)

        mask_en_word = mask_en_en + mask_fr_en - mask_en_en*mask_fr_en
        mask_fr_word = mask_en_fr + mask_fr_fr - mask_en_fr*mask_fr_fr
        """

        unlabeled_loss_function = nn.KLDivLoss(reduction='none')

        # output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        # output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        # loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en)
        #output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en).sum(dim=1)#*mask_en_word.view(-1)
        #loss = loss.sum() / mask_en_word.sum() #(self.batch_size * seq_len_en)
        #if mask_en_word.sum().cpu().numpy() > 1:
        loss = loss.sum() / (self.batch_size * seq_len)
        #else:
        #    loss = loss.sum()

        # output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        output_word_fr_fr = F.log_softmax(output_word_fr_fr, dim=1)
        #output_word_fr_fr = F.log_softmax(SRL_output_fr, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr_fr, output_word_en_fr).sum(dim=1)#*mask_fr_word.view(-1)
        #if mask_fr_word.sum().cpu().numpy() > 1:
        loss_2 = loss_2.sum() / (self.batch_size * seq_len_fr)
        #else:
        #    loss_2 = loss_2.sum()

        #print(mask_en_word.sum())
        #print(mask_fr_word.sum())


        return loss, loss_2, CopyLoss_en, CopyLoss_fr, CopyLoss_en_noise, CopyLoss_fr_noise

    def forward(self, batch_input, lang='En', unlabeled=False, self_constrain=False, use_bert=False, isTrain=False):
        if unlabeled:
            loss, loss_2, copy_loss_en, copy_loss_fr,a,b = self.parallel_train_(batch_input, use_bert)

            return loss, loss_2, copy_loss_en, copy_loss_fr, a, b
            #l2loss = self.word_trans(batch_input, use_bert)
            #return l2loss

        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        predicates_1D = batch_input['predicates_idx']
        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        word_id = get_torch_variable_from_np(batch_input['word_times'])
        word_id_emb = self.id_embedding(word_id)
        flag_emb = self.flag_embedding(flag_batch)
        actual_lens = batch_input['seq_len']

        if use_bert:
            bert_input_ids = get_torch_variable_from_np(batch_input['bert_input_ids'])
            bert_input_mask = get_torch_variable_from_np(batch_input['bert_input_mask'])
            bert_out_positions = get_torch_variable_from_np(batch_input['bert_out_positions'])

            bert_emb = self.model(bert_input_ids, attention_mask=bert_input_mask)
            bert_emb = bert_emb[0]
            bert_emb = bert_emb[:, 1:-1, :].contiguous().detach()

            bert_emb = bert_emb[torch.arange(bert_emb.size(0)).unsqueeze(-1), bert_out_positions].detach()

            for i in range(len(bert_emb)):
                if i >= len(actual_lens):
                    break
                for j in range(len(bert_emb[i])):
                    if j >= actual_lens[i]:
                        bert_emb[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))


            bert_emb = bert_emb.detach()
        #bert_emb = self.bert_NonlinearTrans(bert_emb)
        #bert_emb_noise = gaussian(bert_emb, isTrain, 0, 0.1).detach()

        if lang == "En":
            pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
            #bert_emb = gaussian(bert_emb, isTrain, 0, 0.1).detach()
            #bert_emb = self.En_LinearTrans(bert_emb).detach()
        else:
            pretrain_emb = self.fr_pretrained_embedding(pretrain_batch).detach()
            #bert_emb = self.Fr_LinearTrans(bert_emb).detach()
            #bert_emb = self.Fr2En_Trans(bert_emb).detach()
        #bert_emb = self.bert_FeatureExtractor(bert_emb)
        seq_len = flag_emb.shape[1]
        if not use_bert:
            SRL_output = self.SR_Labeler(pretrain_emb, flag_emb, predicates_1D, seq_len, para=False)

            SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
            SRL_input = SRL_input
            pred_recur = self.SR_Compressor(SRL_input, pretrain_emb,
                                            flag_emb.detach(), word_id_emb, predicates_1D, seq_len, para=False)

            output_word = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), word_id_emb.detach(), seq_len,
                                          para=False)

        else:
            SRL_output = self.SR_Labeler(bert_emb, flag_emb, predicates_1D, seq_len, para=False, use_bert=True)

            SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
            SRL_input_probs = F.softmax(SRL_input, 2).detach()


            if isTrain:
                pred_recur = self.SR_Compressor(SRL_input_probs, bert_emb.detach(),
                                                flag_emb.detach(), word_id_emb, predicates_1D, seq_len, para=False,
                                                use_bert=False)

                output_word = self.SR_Matcher(pred_recur, bert_emb.detach(), flag_emb.detach(), word_id_emb.detach(), seq_len, copy=True,
                                              para=False, use_bert=False)
            else:
                pred_recur = self.SR_Compressor(SRL_input_probs, bert_emb.detach(),
                                                flag_emb.detach(), word_id_emb, predicates_1D, seq_len, para=False,
                                                use_bert=False)

                output_word = self.SR_Matcher(pred_recur, bert_emb.detach(), flag_emb.detach(), word_id_emb.detach(),
                                              seq_len, para=False, use_bert=False)

            score4Null = torch.zeros_like(output_word[:, 1:2])
            output_word = torch.cat((output_word[:, 0:1], score4Null, output_word[:, 1:]), 1)

            teacher = F.softmax(SRL_input.view(self.batch_size * seq_len, -1), dim=1).detach()
            student = F.log_softmax(output_word, dim=1)
            unlabeled_loss_function = nn.KLDivLoss(reduction='none')

            loss_copy = unlabeled_loss_function(student, teacher).view(self.batch_size*seq_len,-1)
            loss_copy = loss_copy.sum() / (self.batch_size*seq_len)

            #CopyLoss = self.learn_loss(SRL_output, pretrain_emb, flag_emb.detach(), seq_len, mask_copy,
            #                              mask_unk)


        return SRL_output, output_word, loss_copy





