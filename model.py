from __future__ import print_function
import numpy as np
import torch
from torch import nn
import sys
#from torch.autograd import Variable
import torch.nn.functional as F
from transformers import *

#from utils import USE_CUDA
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


        self.bilstm_hidden_state_word = (
             torch.zeros(2 * 2, self.batch_size, self.target_vocab_size*10).to(device),
             torch.zeros(2 * 2, self.batch_size, self.target_vocab_size*10).to(device))

        self.bilstm_hidden_state_bert = (
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size * 10).to(device))


        self.bilstm_layer_word = nn.LSTM(input_size=300+self.target_vocab_size+2*self.flag_emb_size,
                                    hidden_size=self.target_vocab_size*10, num_layers=2,
                                    bidirectional=True,
                                    bias=True, batch_first=True)

        self.bilstm_layer_bert = nn.LSTM(input_size=768 + self.target_vocab_size + 1*self.flag_emb_size,
                                         hidden_size=self.target_vocab_size * 10, num_layers=2,
                                         bidirectional=True,
                                         bias=True, batch_first=True)

    def forward(self, SRL_input, word_emb, flag_emb, word_id_emb, predicates_1D, seq_len, use_bert=False, para=False):
        self.bilstm_hidden_state_word = (
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size * 10).to(device))

        self.bilstm_hidden_state_bert = (
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, self.target_vocab_size * 10).to(device))

        SRL_input = SRL_input.view(self.batch_size, seq_len, -1)
        if not use_bert:
            compress_input = torch.cat((word_emb, flag_emb, word_id_emb, SRL_input), 2)
            bilstm_output_word, (_, bilstm_final_state_word) = self.bilstm_layer_word(compress_input,
                                                                                      self.bilstm_hidden_state_word)
            bilstm_output = bilstm_output_word.contiguous()
        else:
            compress_input = torch.cat((word_emb, flag_emb, SRL_input), 2)
            bilstm_output_bert, (_, bilstm_final_state_bert) = self.bilstm_layer_bert(compress_input,
                                                                                      self.bilstm_hidden_state_bert)
            bilstm_output = bilstm_output_bert.contiguous()
        pred_recur = torch.max(bilstm_output, dim=1)[0]
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
        self.compress_word = nn.Sequential(nn.Linear(300+2*self.flag_emb_size, 20), nn.ReLU())
        self.compress_bert = nn.Sequential(nn.Linear(768+0*self.flag_emb_size, 20), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(40, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 1))
        self.match_word = nn.Sequential(nn.Linear(2 * self.bilstm_hidden_size+300+2*self.flag_emb_size, self.mlp_size),
                                        nn.ReLU(),
                                        nn.Linear(self.mlp_size, self.target_vocab_size))


    def forward(self, pred_recur, pretrained_emb, flag_emb, word_id_emb, seq_len, use_bert=False, para=False):
        """
        pred_recur = pred_recur.view(self.batch_size, self.bilstm_hidden_size * 2)
        pred_recur = pred_recur.unsqueeze(1).expand(self.batch_size, seq_len, self.bilstm_hidden_size * 2)
        combine = torch.cat((pred_recur, pretrained_emb, flag_emb, word_id_emb), 2)
        output_word = self.match_word(combine)
        output_word = output_word.view(self.batch_size * seq_len, -1)
        """

        forward_hidden = pred_recur[:, :self.target_vocab_size * 10].view(self.batch_size, self.target_vocab_size,
                                                                             10)
        backward_hidden = pred_recur[:, self.target_vocab_size * 10:].view(self.batch_size, self.target_vocab_size,
                                                                              10)
        role_hidden = torch.cat((forward_hidden, backward_hidden), 2)
        role_hidden = role_hidden.unsqueeze(1).expand(self.batch_size, seq_len, self.target_vocab_size, 20)
        if not use_bert:
            combine = self.compress_word(torch.cat((pretrained_emb,  word_id_emb), 2))
        else:
            combine = self.compress_bert(pretrained_emb)
        combine = combine.unsqueeze(2).expand(self.batch_size, seq_len, self.target_vocab_size, 20)
        scores = self.scorer(torch.cat((role_hidden, combine), 3)).view(self.batch_size*seq_len, self.target_vocab_size)

        return scores




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
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()

    def role_coverage(self, output_en, output_fr):
        _, roles_en = torch.max(output_en, 2)
        _, roles_fr = torch.max(output_fr, 2)
        coverages = [0.0]*self.batch_size
        for i in range(self.batch_size):
            NonNull_Truth = 0.0
            NonNull_Predict = 0.0
            In_NonNull_Predict = 0.0
            for role in roles_en[i]:
                if role > 1:
                    NonNull_Truth += 1
            if NonNull_Truth < 1:
                coverages[i] = 0.0
                continue
            for role in roles_fr[i]:
                if role > 1:
                    NonNull_Predict += 1
                    if role in roles_en[i]:
                        In_NonNull_Predict += 1
            if In_NonNull_Predict == 0 or NonNull_Predict==0:
                coverages[i] = 0.0
                continue
            P = In_NonNull_Predict/NonNull_Predict
            R = In_NonNull_Predict/NonNull_Truth
            F = 2 * P * R / (P + R)
            coverages[i] = F

        return coverages

    def copy_loss(self, output_SRL, flag_emb, pretrain_emb, seq_len):
        SRL_input = output_SRL.view(self.batch_size, seq_len, -1)
        SRL_input = SRL_input.detach()
        pred_recur = self.SR_Compressor(SRL_input, pretrain_emb,
                                        flag_emb.detach(), None, None, seq_len, para=False, use_bert=True)

        output_word = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), None, seq_len,
                                      para=False, use_bert=True)
        teacher = F.softmax(output_SRL.view(self.batch_size*seq_len, -1), dim=1).detach()
        student = F.log_softmax(output_word.view(self.batch_size*seq_len, -1), dim=1)
        unlabeled_loss_function = nn.KLDivLoss(reduction='none')
        loss = unlabeled_loss_function(student, teacher)
        loss = loss.sum() / (self.batch_size * seq_len)
        return loss

    #for En sentence, loss I
    #if Fr event vector find an En word which could output same argument with an Fr word,
    # think this En word should be Ax. Then, we need to ask the opinion of En event vector about this En word
    def P_word_mask(self, output_fr_en, output_fr_fr, seq_len_en):
        word_mask = np.zeros((self.batch_size, seq_len_en), dtype="float32")

        _, roles_fr_en = torch.max(output_fr_en, 2)
        _, roles_fr_fr = torch.max(output_fr_fr, 2)
        for i in range(self.batch_size):
            fr_roles_set = []
            for role in roles_fr_fr[i]:
                if role not in fr_roles_set:
                    fr_roles_set.append(role)
            for j in range(seq_len_en):
                if roles_fr_en[i][j] < 2:
                    continue
                elif roles_fr_en[i][j] in fr_roles_set:
                    word_mask[i][j] = 1.0
        return word_mask

    def word_mask(self,output_en_en, output_en_fr, seq_len_en, seq_len_fr):
        word_mask_en = np.zeros((self.batch_size, seq_len_en), dtype="float32")
        word_mask_fr = np.zeros((self.batch_size, seq_len_fr), dtype="float32")
        _, roles_en_fr = torch.max(output_en_fr, 2)
        _, roles_en_en = torch.max(output_en_en, 2)
        for i in range(self.batch_size):
            en_role_set = [-1]*self.target_vocab_size

            all_null = True
            for id, role in enumerate(roles_en_en[i]):
                if role > 1:
                    all_null = False
                    en_role_set[role] = id
            fr_role_set = [-1] * self.target_vocab_size
            if all_null:
                continue

            found_already = False
            for id, role in enumerate(roles_en_fr[i]):
                if role > 1:
                    if fr_role_set[role] != -1:
                        found_already=True
                        break
                    fr_role_set[role] = id
            if found_already:
                continue

            for a, b in zip(en_role_set[2:], fr_role_set[2:]):
                if a==-1 and b>=0:
                    found_already = True
                    break
                if a>=0 and b==-1:
                    found_already = True
                    break
            if found_already:
                continue

            word_mask_en[i] = np.ones((seq_len_en,), dtype="float32")
            word_mask_fr[i] = np.ones((seq_len_fr,), dtype="float32")

            """
            for id in en_role_set:
                if id!=-1:
                    word_mask_en[i][id] = 1.0
            for id in fr_role_set:
                if id != -1:
                    word_mask_fr[i][id] = 1.0
            """
        return word_mask_en, word_mask_fr

    def word_mask_soft(self,output_en_en, output_en_fr, seq_len_en, seq_len_fr):
        word_mask_en = np.zeros((self.batch_size, seq_len_en), dtype="float32")
        word_mask_fr = np.zeros((self.batch_size, seq_len_fr), dtype="float32")
        _, roles_en_fr = torch.max(output_en_fr, 2)
        _, roles_en_en = torch.max(output_en_en, 2)
        for i in range(self.batch_size):
            en_role_set = [-1]*self.target_vocab_size
            en_role_num = 0
            for id, role in enumerate(roles_en_en[i]):
                if role > 1:
                    en_role_set[role] = id
                    en_role_num += 1

            fr_role_num = 0
            co_role_num = 0
            fr_role_set = [-1] * self.target_vocab_size
            for id, role in enumerate(roles_en_fr[i]):
                if role > 1:
                    fr_role_set[role] = id
                    fr_role_num += 1
                    if en_role_set[role] >=0:
                        co_role_num += 1


            if en_role_num > 1 and co_role_num ==0:
                continue

            word_mask_en[i] = np.ones((seq_len_en,), dtype="float32")
            word_mask_fr[i] = np.ones((seq_len_fr,), dtype="float32")

            """
            for id in en_role_set:
                if id!=-1:
                    word_mask_en[i][id] = 1.0
            for id in fr_role_set:
                if id != -1:
                    word_mask_fr[i][id] = 1.0
            """
        return word_mask_en, word_mask_fr



    # for Fr sentence, loss II
    # if En event vector find an Fr word which could output same argument with an En word,
    # think this Fr word should be Ax. Then, we need to check the opinion of Fr event vector about this Fr word
    def R_word_mask(self, output_en_en, output_en_fr, seq_len_fr):
        word_mask = np.zeros((self.batch_size, seq_len_fr), dtype="float32")

        _, roles_en_en = torch.max(output_en_en, 2)
        _, roles_en_fr = torch.max(output_en_fr, 2)
        for i in range(self.batch_size):
            en_roles_set = []
            for role in roles_en_en[i]:
                if role not in en_roles_set:
                    en_roles_set.append(role)
            for j in range(seq_len_fr):
                if roles_en_fr[i][j] < 2:
                    continue
                elif roles_en_fr[i][j] in en_roles_set:
                    word_mask[i][j] = 1.0
        return word_mask

    def parallel_train(self, batch_input, use_bert, isTrain=True):
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
            bert_emb_fr = gaussian(bert_emb_fr, isTrain, 0, 0.1)
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
            bert_emb_en = gaussian(bert_emb_en, isTrain, 0, 0.1)
            bert_emb_en = bert_emb_en.detach()


        seq_len = flag_emb.shape[1]
        SRL_output = self.SR_Labeler(bert_emb_en, flag_emb.detach(), predicates_1D, seq_len, para=True, use_bert=True)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        pred_recur = self.SR_Compressor(SRL_input.detach(), bert_emb_en,
                                        flag_emb.detach(), None, predicates_1D, seq_len, para=True, use_bert=True)



        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(bert_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True, use_bert=True)

        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        pred_recur_fr = self.SR_Compressor(SRL_input_fr, bert_emb_fr,
                                        flag_emb_fr.detach(), None, predicates_1D_fr, seq_len_fr, para=True, use_bert=True)

        """
        En event vector, En word
        """
        output_word_en_en = self.SR_Matcher(pred_recur.detach(), bert_emb_en, flag_emb.detach(), None, seq_len,
                                         para=True, use_bert=True).detach()

        #############################################
        """
        Fr event vector, En word
        """
        output_word_fr_en = self.SR_Matcher(pred_recur_fr, bert_emb_en, flag_emb.detach(), None, seq_len,
                                      para=True, use_bert=True)

        ## B*T R 2
        Union_enfr_en = torch.cat((output_word_en_en.view(-1, self.target_vocab_size, 1),
                                   output_word_fr_en.view(-1, self.target_vocab_size, 1)), 2)
        ## B*T R
        max_enfr_en = torch.max(Union_enfr_en, 2)[0]

        #############################################3
        """
        En event vector, Fr word
        """
        output_word_en_fr = self.SR_Matcher(pred_recur.detach(), bert_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                         para=True, use_bert=True).detach()
        """
        Fr event vector, Fr word
        """
        output_word_fr_fr = self.SR_Matcher(pred_recur_fr, bert_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                         para=True, use_bert=True)

        ## B*T R 2
        Union_enfr_fr = torch.cat((output_word_en_fr.view(-1, self.target_vocab_size, 1),
                                   output_word_fr_fr.view(-1, self.target_vocab_size, 1)), 2)
        ## B*T R
        max_enfr_fr = torch.max(Union_enfr_fr, 2)[0]

        unlabeled_loss_function = nn.KLDivLoss(reduction='none')
        """
        word_mask_4en = self.P_word_mask(output_word_fr_en.view(self.batch_size, seq_len, -1),
                                         output_word_fr_fr.view(self.batch_size, seq_len_fr, -1), seq_len_en)
        word_mask_4en_tensor = get_torch_variable_from_np(word_mask_4en).view(self.batch_size*seq_len_en, -1)

        word_mask_4fr = self.R_word_mask(output_word_en_en.view(self.batch_size, seq_len, -1),
                                         output_word_en_fr.view(self.batch_size, seq_len_fr, -1), seq_len_fr)
        word_mask_4fr_tensor = get_torch_variable_from_np(word_mask_4fr).view(self.batch_size*seq_len_fr, -1)
        
        word_mask_en, word_mask_fr = self.word_mask_soft(output_word_en_en.view(self.batch_size, seq_len_en, -1),
                                                    output_word_en_fr.view(self.batch_size, seq_len_fr, -1),
                                                    seq_len_en, seq_len_fr)
        word_mask_en_tensor = get_torch_variable_from_np(word_mask_en).view(self.batch_size * seq_len_en)
        word_mask_fr_tensor = get_torch_variable_from_np(word_mask_fr).view(self.batch_size * seq_len_fr)
        """
        #output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        #output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        #loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en)
        output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        max_enfr_en = F.log_softmax(max_enfr_en, dim=1)
        loss = unlabeled_loss_function(max_enfr_en, output_word_en_en)
        loss = loss.sum(dim=1)#*word_mask_en_tensor
        loss = loss.sum() / (self.batch_size*seq_len_en)

        #output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        max_enfr_fr = F.softmax(max_enfr_fr, dim=1).detach()
        output_word_fr_fr = F.log_softmax(output_word_fr_fr, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr_fr, max_enfr_fr)
        loss_2 = loss_2.sum(dim=1)#*word_mask_fr_tensor

        loss_2 = loss_2.sum()/ (self.batch_size*seq_len_fr)

        return  loss, loss_2

    """
    def self_train_hidden(self, batch_input):
        unlabeled_data_en, unlabeled_data_fr = batch_input

        pretrain_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['pretrain'])
        predicates_1D_fr = unlabeled_data_fr['predicates_idx']
        flag_batch_fr = get_torch_variable_from_np(unlabeled_data_fr['flag'])
        flag_emb_fr = self.flag_embedding(flag_batch_fr).detach()
        pretrain_emb_fr = self.fr_pretrained_embedding(pretrain_batch_fr).detach()

        pretrain_batch = get_torch_variable_from_np(unlabeled_data_en['pretrain'])
        predicates_1D = unlabeled_data_en['predicates_idx']
        flag_batch = get_torch_variable_from_np(unlabeled_data_en['flag'])
        flag_emb = self.flag_embedding(flag_batch)
        seq_len = flag_emb.shape[1]
        pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()

        SRL_output = self.SR_Labeler(pretrain_emb, flag_emb.detach(), predicates_1D, seq_len, para=True).view(self.batch_size, seq_len, -1)
        SRL_input = F.softmax(SRL_output, 2)
        # B R
        max_role_en = torch.max(SRL_input, dim=1)[0].detach()

        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(pretrain_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True).view(self.batch_size, seq_len_fr, -1)
        SRL_input_fr = F.softmax(SRL_output_fr, 2)
        # B R
        max_role_fr = torch.max(SRL_input_fr, dim=1)[0].detach()
        criterion = nn.MSELoss(size_average=False)
        loss = criterion(max_role_fr, max_role_en) / self.batch_size
        return loss

    def self_train(self, batch_input):
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
        pred_recur = self.SR_Compressor(SRL_input.detach(), pretrain_emb,
                                        flag_emb.detach(), word_id_emb.detach(), predicates_1D, seq_len, para=True)
        seq_len_fr = flag_emb_fr.shape[1]

     
        output_word_en = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), word_id_emb.detach(),
                                         seq_len,
                                         para=True)
        output_word_en = output_word_en.view(self.batch_size, seq_len, self.target_vocab_size)

        output_word_en = F.softmax(output_word_en, 2)
        # B R
        max_role_en = torch.max(output_word_en, dim=1)[0].detach()

        output_word_fr= self.SR_Matcher(pred_recur, pretrain_emb_fr, flag_emb_fr.detach(),
                                         word_id_emb_fr.detach(), seq_len_fr,
                                         para=True)
        output_word_fr = output_word_fr.view(self.batch_size, seq_len_fr, self.target_vocab_size)
        output_word_fr = F.softmax(output_word_fr, 2)
        # B R
        max_role_fr = torch.max(output_word_fr, dim=1)[0]
        criterion = nn.MSELoss(size_average=False)
        loss = criterion(max_role_fr, max_role_en)/self.batch_size

        return loss

    """


    def forward(self, batch_input, lang='En', unlabeled=False, self_constrain=False, use_bert=False, isTrain=False):
        if unlabeled:

            loss = self.parallel_train(batch_input, use_bert)

            loss_word = 0

            return loss, loss_word
        if self_constrain:

            loss = self.self_train(batch_input)

            return loss

        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])
        predicates_1D = batch_input['predicates_idx']
        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        word_id = get_torch_variable_from_np(batch_input['word_times'])
        word_id_emb = self.id_embedding(word_id)
        flag_emb = self.flag_embedding(flag_batch)
        actual_lens = batch_input['seq_len']
        #print(actual_lens)
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


            bert_emb = gaussian(bert_emb, isTrain, 0, 0.1)
            bert_emb = bert_emb.detach()

        if lang == "En":
            pretrain_emb = self.pretrained_embedding(pretrain_batch).detach()
        else:
            pretrain_emb = self.fr_pretrained_embedding(pretrain_batch).detach()




        seq_len = flag_emb.shape[1]
        if not use_bert:
            SRL_output = self.SR_Labeler(pretrain_emb, flag_emb, predicates_1D, seq_len, para=False)


            SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
            SRL_input = SRL_input
            pred_recur = self.SR_Compressor(SRL_input, pretrain_emb,
                                            flag_emb.detach(), word_id_emb, predicates_1D, seq_len, para=False)

            output_word = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), word_id_emb.detach(), seq_len, para=False)

        else:
            SRL_output = self.SR_Labeler(bert_emb, flag_emb, predicates_1D, seq_len, para=False, use_bert=True)

            SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
            SRL_input = SRL_input
            pred_recur = self.SR_Compressor(SRL_input, bert_emb,
                                            flag_emb.detach(), word_id_emb, predicates_1D, seq_len, para=False,use_bert=True)

            output_word = self.SR_Matcher(pred_recur, bert_emb, flag_emb.detach(), word_id_emb.detach(), seq_len,
                                          para=False, use_bert=True)

        return SRL_output, output_word






