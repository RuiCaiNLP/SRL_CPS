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

        self.bilstm_layer_bert = nn.LSTM(input_size=768 + self.target_vocab_size + self.flag_emb_size,
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
        self.compress_bert = nn.Sequential(nn.Linear(768, 20), nn.ReLU())
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
            combine = self.compress_word(torch.cat((pretrained_emb, flag_emb, word_id_emb), 2))
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
        self.model.eval()



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
        pred_recur = self.SR_Compressor(SRL_input.detach(), pretrain_emb,
                                        flag_emb.detach(), word_id_emb.detach(), predicates_1D, seq_len, para=True)



        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(pretrain_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True)

        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        pred_recur_fr = self.SR_Compressor(SRL_input_fr, pretrain_emb_fr,
                                        flag_emb_fr.detach(), word_id_emb_fr, predicates_1D_fr, seq_len_fr, para=True)

        #L2_loss_function = nn.MSELoss(size_average=False)
        #l2_loss = L2_loss_function(pred_recur_fr, pred_recur.detach())/self.batch_size
        """
        En event vector, En word
        """
        output_word_en = self.SR_Matcher(pred_recur.detach(), pretrain_emb, flag_emb.detach(), word_id_emb.detach(), seq_len,
                                         para=True)

        #############################################
        """
        Fr event vector, En word
        """
        output_word_fr = self.SR_Matcher(pred_recur_fr, pretrain_emb, flag_emb.detach(), word_id_emb.detach(), seq_len,
                                      para=True)
        unlabeled_loss_function = nn.KLDivLoss(size_average=False)
        output_word_en = F.softmax(output_word_en, dim=1).detach()
        output_word_fr = F.log_softmax(output_word_fr, dim=1)
        loss = unlabeled_loss_function(output_word_fr, output_word_en)/(seq_len_en*self.batch_size)
        #############################################3
        """
        En event vector, Fr word
        """
        output_word_en = self.SR_Matcher(pred_recur.detach(), pretrain_emb_fr, flag_emb_fr.detach(), word_id_emb_fr.detach(), seq_len_fr,
                                         para=True)
        """
        Fr event vector, Fr word
        """
        output_word_fr = self.SR_Matcher(pred_recur_fr, pretrain_emb_fr, flag_emb_fr.detach(), word_id_emb_fr.detach(), seq_len_fr,
                                         para=True)
        unlabeled_loss_function = nn.KLDivLoss(size_average=False)
        output_word_en = F.softmax(output_word_en, dim=1).detach()
        output_word_fr = F.log_softmax(output_word_fr, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr, output_word_en) / (seq_len_fr*self.batch_size)
        return  loss, loss_2

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

        """
        En event vector, En word
        """
        output_word_en = self.SR_Matcher(pred_recur, pretrain_emb, flag_emb.detach(), word_id_emb.detach(),
                                         seq_len,
                                         para=True)
        output_word_en = output_word_en.view(self.batch_size, seq_len, self.target_vocab_size)

        output_word_en = F.softmax(output_word_en, 2)
        # B R
        max_role_en = torch.max(output_word_en, dim=1)[0].detach()


        """
        En event vector, Fr word
        """
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




    def forward(self, batch_input, lang='En', unlabeled=False, self_constrain=False, use_bert=False):
        if unlabeled:

            loss = self.parallel_train(batch_input)

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
        if use_bert:
            bert_input_ids = get_torch_variable_from_np(batch_input['bert_input_ids'])
            bert_input_mask = get_torch_variable_from_np(batch_input['bert_input_mask'])
            bert_emb = self.model(bert_input_ids, attention_mask=bert_input_mask)
            bert_emb = bert_emb[0]
            bert_emb = bert_emb[:, 1:-1, :].contiguous().detach()

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






