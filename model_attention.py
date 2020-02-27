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

        self.bilstm_hidden_state_word = (
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device))

        self.bilstm_hidden_state_bert = (
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device))

        self.bilstm_layer_word = nn.LSTM(input_size=300 + self.target_vocab_size + 2 * self.flag_emb_size,
                                         hidden_size=self.target_vocab_size * 10, num_layers=2,
                                         bidirectional=True,
                                         bias=True, batch_first=True)

        self.bilstm_layer_bert = nn.LSTM(input_size=256 + self.target_vocab_size + 0 * self.flag_emb_size,
                                         hidden_size=(self.target_vocab_size-1) * 10, num_layers=2,
                                         bidirectional=True,
                                         bias=True, batch_first=True)

    def forward(self, SRL_input, word_emb, flag_emb, word_id_emb, predicates_1D, seq_len, use_bert=False, para=False):
        self.bilstm_hidden_state_word = (
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device))

        self.bilstm_hidden_state_bert = (
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device),
            torch.zeros(2 * 2, self.batch_size, (self.target_vocab_size-1) * 10).to(device))

        SRL_input = SRL_input.view(self.batch_size, seq_len, -1)
        if not use_bert:
            compress_input = torch.cat((word_emb, flag_emb, word_id_emb, SRL_input), 2)
            bilstm_output_word, (_, bilstm_final_state_word) = self.bilstm_layer_word(compress_input,
                                                                                      self.bilstm_hidden_state_word)
            bilstm_output = bilstm_output_word.contiguous()
        else:
            compress_input = torch.cat((word_emb, SRL_input), 2)
            bilstm_output_bert, (_, bilstm_final_state_bert) = self.bilstm_layer_bert(compress_input,
                                                                                      self.bilstm_hidden_state_bert)
            bilstm_output = bilstm_output_bert.contiguous()
        pred_recur = torch.max(bilstm_output, dim=1)[0]
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
        self.bert_size = 768
        self.base_emb2vector = nn.Sequential(nn.Linear(self.bert_size, 300),
                                        nn.ReLU(),
                                        nn.Linear(300, 200),
                                        nn.ReLU())

        self.query_emb2vector = nn.Sequential(nn.Linear(self.bert_size, 300),
                                              nn.ReLU(),
                                              nn.Linear(300, 200),
                                              nn.ReLU())

        self.matrix = nn.Parameter(
            get_torch_variable_from_np(np.zeros((200, 200)).astype("float32")))

    def forward(self, base_embs, query_embs, SRL_scores,  seq_len,  isTrain = False, para=False):
        query_vectors = self.query_emb2vector(query_embs).view(self.batch_size, seq_len, 200)
        base_vectors = self.base_emb2vector(base_embs).view(self.batch_size, seq_len, 200)
        SRL_probs = SRL_scores.view(self.batch_size, seq_len, self.target_vocab_size)
        base_vectors = base_vectors.view(self.batch_size * seq_len, 200)
        y = torch.mm(base_vectors, self.matrix)
        # B T2 V -> B V T2
        query_vectors = query_vectors.transpose(1, 2).contiguous()
        # B T1 v * B v T2 -> B T1 T2
        scores = torch.bmm(y.view(self.batch_size, seq_len, 200), query_vectors)
        scores = scores.transpose(1, 2).contiguous()
        # B T2 T1
        scores = F.softmax(scores, 2)
        # B T2 T1 * B T1 R -> B T2 R
        output_word = torch.bmm(scores, SRL_scores).view(self.batch_size * seq_len, -1)
        return output_word

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

        CopyLoss_en = self.copy_loss(SRL_output, bert_emb_en, flag_emb.detach(), seq_len)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        SRL_input = F.softmax(SRL_input, 2)
        pred_recur = self.SR_Compressor(SRL_input.detach(), bert_emb_en_noise,
                                        flag_emb.detach(), None, predicates_1D, seq_len, para=True, use_bert=True)

        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(bert_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True,
                                        use_bert=True)

        CopyLoss_fr = self.copy_loss(SRL_output_fr, bert_emb_fr_noise, flag_emb_fr.detach(), seq_len_fr)


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





        unlabeled_loss_function = nn.KLDivLoss(reduction='none')

        # output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        # output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        # loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en)
        output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()
        output_word_fr_en = F.log_softmax(output_word_fr_en, dim=1)
        loss = unlabeled_loss_function(output_word_fr_en, output_word_en_en)
        loss = loss.sum() / (self.batch_size * seq_len_en)

        # output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()
        output_word_fr_fr = F.log_softmax(output_word_fr_fr, dim=1)
        loss_2 = unlabeled_loss_function(output_word_fr_fr, output_word_en_fr)
        loss_2 = loss_2.sum() / (self.batch_size * seq_len_fr)



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
            # bert_emb_fr = gaussian(bert_emb_fr, isTrain, 0, 0.1)
            bert_emb_fr = bert_emb_fr.detach()

            bert_input_ids_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_ids'])
            bert_input_mask_en = get_torch_variable_from_np(unlabeled_data_en['bert_input_mask'])
            bert_out_positions_en = get_torch_variable_from_np(unlabeled_data_en['bert_out_positions'])

            bert_emb_en = self.model(bert_input_ids_en, attention_mask=bert_input_mask_en)
            bert_emb_en = bert_emb_en[0]
            # bert_emb_en = bert_emb_en[:, 1:-1, :].contiguous().detach()
            bert_emb_en = bert_emb_en[torch.arange(bert_emb_en.size(0)).unsqueeze(-1), bert_out_positions_en].detach()

            for i in range(len(bert_emb_en)):
                if i >= len(actual_lens_en):
                    print("error")
                    break
                for j in range(len(bert_emb_en[i])):
                    if j >= actual_lens_en[i]:
                        bert_emb_en[i][j] = get_torch_variable_from_np(np.zeros(768, dtype="float32"))
            # bert_emb_en = gaussian(bert_emb_en, isTrain, 0, 0.1)
            bert_emb_en = bert_emb_en.detach()

        seq_len = flag_emb.shape[1]
        SRL_output = self.SR_Labeler(bert_emb_en, flag_emb.detach(), predicates_1D, seq_len, para=True, use_bert=True)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        pred_recur = self.SR_Compressor(SRL_input.detach(), bert_emb_en,
                                        flag_emb.detach(), None, predicates_1D, seq_len, para=True, use_bert=True)

        seq_len_fr = flag_emb_fr.shape[1]
        SRL_output_fr = self.SR_Labeler(bert_emb_fr, flag_emb_fr.detach(), predicates_1D_fr, seq_len_fr, para=True,
                                        use_bert=True)

        SRL_input_fr = SRL_output_fr.view(self.batch_size, seq_len_fr, -1)
        pred_recur_fr = self.SR_Compressor(SRL_input_fr, bert_emb_fr,
                                           flag_emb_fr.detach(), None, predicates_1D_fr, seq_len_fr, para=True,
                                           use_bert=True)

        """
        En event vector, En word
        """
        output_word_en_en = self.SR_Matcher(pred_recur.detach(), bert_emb_en, flag_emb.detach(), None, seq_len,
                                            para=True, use_bert=True)
        output_word_en_en = F.softmax(output_word_en_en, dim=1).detach()

        score4Null = torch.zeros_like(output_word_en_en[:, 1:2])
        output_word_en_en = torch.cat((output_word_en_en[:, 0:1], score4Null, output_word_en_en[:, 1:]), 1)


        #############################################
        """
        Fr event vector, En word
        """
        output_word_fr_en = self.SR_Matcher(pred_recur_fr, bert_emb_en, flag_emb.detach(), None, seq_len,
                                            para=True, use_bert=True)
        output_word_fr_en = F.softmax(output_word_fr_en, dim=1)

        score4Null = torch.zeros_like(output_word_fr_en[:, 1:2])
        output_word_fr_en = torch.cat((output_word_fr_en[:, 0:1], score4Null, output_word_fr_en[:, 1:]), 1)

        #############################################3
        """
        En event vector, Fr word
        """
        output_word_en_fr = self.SR_Matcher(pred_recur.detach(), bert_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                            para=True, use_bert=True)
        output_word_en_fr = F.softmax(output_word_en_fr, dim=1).detach()

        score4Null = torch.zeros_like(output_word_en_fr[:, 1:2])
        output_word_en_fr = torch.cat((output_word_en_fr[:, 0:1], score4Null, output_word_en_fr[:, 1:]), 1)

        """
        Fr event vector, Fr word
        """
        output_word_fr_fr = self.SR_Matcher(pred_recur_fr, bert_emb_fr, flag_emb_fr.detach(), None, seq_len_fr,
                                            para=True, use_bert=True)
        output_word_fr_fr = F.softmax(output_word_fr_fr, dim=1)

        score4Null = torch.zeros_like(output_word_fr_fr[:, 1:2])
        output_word_fr_fr = torch.cat((output_word_fr_fr[:, 0:1], score4Null, output_word_fr_fr[:, 1:]), 1)

        ## B*T R 2
        # Union_enfr_fr = torch.cat((output_word_en_fr.view(-1, self.target_vocab_size, 1),
        #                           output_word_fr_fr.view(-1, self.target_vocab_size, 1)), 2)
        ## B*T R
        # max_enfr_fr = torch.max(output_word_fr_fr, output_word_en_fr).detach()
        # max_enfr_fr[:, :2] = output_word_fr_fr[:,:2].detach()


        unlabeled_loss_function = nn.L1Loss(reduction='none')
        loss = unlabeled_loss_function(output_word_fr_en[:, 1], output_word_en_en[:, 1])
        theta = torch.gt(output_word_en_en[:, 1], output_en_en_nonNull_max)
        loss = theta * loss
        if torch.gt(theta.sum(), 0):
            loss = loss.sum() / theta.sum()
        else:
            loss = loss.sum()

        loss_2 = unlabeled_loss_function(output_word_fr_fr_nonNull_maxarg.view(-1), output_en_fr_nonNull_max)
        theta = torch.gt(output_en_fr_nonNull_max, output_word_en_fr[:, 1])
        loss_2 = theta * loss_2
        if torch.gt(theta.sum(), 0):
            loss_2 = loss_2.sum() / theta.sum()
        else:
            loss_2 = loss_2.sum()

        return loss, loss_2

    def forward(self, batch_input, lang='En', unlabeled=False, self_constrain=False, use_bert=False, isTrain=False):
        if unlabeled:
            #l2loss = self.word_trans(batch_input, use_bert)
            consistent_loss = self.parallel_train(batch_input, use_bert)
            return consistent_loss
        predicates_1D = batch_input['predicates_idx']
        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        flag_emb = self.flag_embedding(flag_batch)
        actual_lens = batch_input['seq_len']
        # print(actual_lens)
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

        seq_len = flag_emb.shape[1]

        SRL_output = self.SR_Labeler(bert_emb, flag_emb, predicates_1D, seq_len, para=False, use_bert=True)

        SRL_input = SRL_output.view(self.batch_size, seq_len, -1)
        SRL_input_scores = SRL_input.detach()


        output_word = self.SR_Matcher(bert_emb.detach(), bert_emb.detach(), SRL_input_scores,
                                      seq_len, isTrain=isTrain,  para=False)

        #score4Null = torch.zeros_like(output_word[:, 1:2])
        #output_word = torch.cat((output_word[:, 0:1], score4Null, output_word[:, 1:]), 1)

        teacher = F.softmax(SRL_input.view(self.batch_size * seq_len, -1), dim=1).detach()
        student = F.log_softmax(output_word, dim=1)
        unlabeled_loss_function = nn.KLDivLoss(reduction='none')
        loss_copy = unlabeled_loss_function(student, teacher)
        loss_copy = loss_copy.sum() / (self.batch_size * seq_len)

        return SRL_output, output_word, loss_copy






