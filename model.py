from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from model_utils import GraphConvolution, GraphAttentionLayer, MultiNonLinearClassifier,masked_avgpool,gen_adj
from transformers import BertPreTrainedModel, BertModel

class RR_GAT(BertPreTrainedModel):

    def __init__(self, config, params):
        super().__init__(config)
        self.max_text_len = params.max_seq_length
        self.rel_num = params.rel_num
        self.bert = BertModel(config)

        self.EGC = GraphConvolution(config.hidden_size*2, 768)
        self.ent_gat = GraphAttentionLayer(768, 768, 0.5, 0.2, True)
        # self.ent_gat = GAT(nfeat=100, nhid=8, nheads=8, nclass=100, alpha=0.2, dropout=0.5)
        # relation classification
        self.rel_judgement = MultiNonLinearClassifier(768, 24, 0.5)  
        self.rel_embedding = nn.Embedding(24, 768)
        self.seq_tag_size = params.seq_tag_size

        self.sequence_tagging_sub = MultiNonLinearClassifier(768, 3, 0.5)
        self.sequence_tagging_obj = MultiNonLinearClassifier(768, 3, 0.5)
        self.global_corres = MultiNonLinearClassifier(768*2, 1, 0.5)
        self.init_weights()
        self.dev = params.device

    def forward(self, input_ids=None, 
                attention_mask=None, 
                edge_matrix=None, 
                seq_tags=None,
                potential_rels=None,
                # corres_tags=None,
                rel_tags=None,
                subs=None,
                objs=None):
    
        corres_threshold, rel_threshold = 0.5, 0.5
        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        # pool_output = outputs[1]  # [32, 768]
        bs, seq_len, h = sequence_output.size()  # 32,100,768
        # 通过非线性多分类得到关系预测结果和关系embedding
        h_k_avg = masked_avgpool(sequence_output, attention_mask) # [32, 768]
        rel_pred = self.rel_judgement(h_k_avg)              # [32, 24]
        
        # for every position $i$ in sequence, should concate $j$ to predict.
        sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
        obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
        # batch x seq_len x seq_len x 2*hidden
        corres_pred = torch.cat([sub_extend, obj_extend], 3)
        # (bs, seq_len, seq_len)
        corres_pred = self.global_corres(corres_pred).squeeze(-1)
        mask_tmp1 = attention_mask.unsqueeze(-1)
        mask_tmp2 = attention_mask.unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if seq_tags is None:
            # (bs, rel_num)
            rel_threshold = 0.5
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))

            # if potential relation is null
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample

            # 2*(sum(x_i),)
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)]

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)

        
        rel_emb = self.rel_embedding(potential_rels)
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)    # [32, 100, 768]
        decode_input = torch.cat([sequence_output, rel_emb], dim=-1)  # 语义和关系作为特征输入， 进行s,o训练  # [32, 100, 1536]

        # 接下进行s,o的学习预测
        # train
        Aj_matrix_1_gen = []
        for i in range(bs):
            # print(i)
            Aj_matrix_1_gen.append(gen_adj(edge_matrix[i]))  # 邻接矩阵
        
        EGC_out = self.EGC(decode_input, Aj_matrix_1_gen)  #edge matrix  list 32 [100, 100]

        # 结合RIFRE，进行实体位置预测
        # 先预测s, 再根据s，GAT结果进行o预测
        Ent_out = torch.stack(EGC_out)
        adj = torch.stack(Aj_matrix_1_gen)
        gat_out = []
        for i in range(bs):
          g_o = self.ent_gat(Ent_out[i], adj[i])
          gat_out.append(g_o)
        gat_out_t = torch.stack(gat_out)
        output_sub = self.sequence_tagging_sub(gat_out_t)
        output_obj = self.sequence_tagging_obj(gat_out_t)

        if seq_tags is not None:
            # calculate loss
            attention_mask = attention_mask.view(-1)
            # sequence label loss
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                      seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                      seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2
            # init
            loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
            
            corres_tags_ = []
            corres_tag = np.zeros((100, 100))
            for b in range(bs):
              heads_s = [i[0] for i in subs[b]]
              heads_o = [i[0] for i in objs[b]]
              for i, j in zip(heads_s, heads_o):
                corres_tag[i][j] = 1
              corres_tags_.append(corres_tag)
            corres_tags = torch.tensor([f for f in corres_tags_], dtype=torch.long)  # [bs, 100, 100]
            corres_tags = corres_tags.to(self.dev)
            corres_pred = corres_pred.view(bs, -1)
            corres_mask = corres_mask.view(bs, -1)
            corres_tags = corres_tags.view(bs, -1)  # [bs, 10000]
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss_matrix = (loss_func(corres_pred,
                          corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

            loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            loss_rel = loss_func(rel_pred, rel_tags.float()) 

            loss = loss_seq + loss_matrix + loss_rel
            return loss, loss_seq, loss_matrix, loss_rel
        else:
            # (sum(x_i), seq_len)
            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            # (sum(x_i), 2, seq_len)
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
            # if ensure_corres:
            corres_pred = torch.sigmoid(corres_pred) * corres_mask
            # (bs, seq_len, seq_len)
            pred_corres_onehot = torch.where(corres_pred > corres_threshold,
                                              torch.ones(corres_pred.size(), device=corres_pred.device),
                                              torch.zeros(corres_pred.size(), device=corres_pred.device))
            return pred_seqs, pred_corres_onehot, xi, pred_rels


if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))
    model = RR_GAT.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    model.to(params.device)

    for n, _ in model.named_parameters(): 
        print(n)