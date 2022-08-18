import torch.nn as nn
from config import Config
import torch
config = Config()
if config.pretrainning_model == 'nezha':
    from NEZHA.model_nezha import BertPreTrainedModel, NEZHAModel
elif config.pretrainning_model == 'albert':
    from transformers import AlbertModel, BertPreTrainedModel
else:
    # bert,roberta
    from transformers import BertPreTrainedModel, BertModel


class BertForCLS(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.params = params
        self.config = config
        # 预训练模型
        if params.pretrainning_model == 'nezha':  # batch_size, max_len, 768
            self.bert = NEZHAModel(config)
        elif params.pretrainning_model == 'albert':
            self.bert = AlbertModel(config)
        else:
            # self.bert = RobertaModel(config)
            self.bert = BertModel(config)

        #  动态权重组件
        self.classifier = nn.Linear(config.hidden_size, 1)  # for dym's dense
        self.dym_pool = nn.Linear(params.embed_dense, 1)  # for dym's dense
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, params.embed_dense),
                                         nn.ReLU(True))  # 动态最后的维度
        self.dense_emb_size = nn.Sequential(nn.Linear(config.hidden_size, params.embed_dense),
                                         nn.ReLU(True))  # 降维
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)
        # self.pool_weight = nn.Parameter(torch.ones((params.batch_size, 1, 1, 1)),
        #                                 requires_grad=True)

        # 下游结构组件
        if params.model_type == 'bilstm':
            num_layers = params.rnn_num
            lstm_num = int(self.params.embed_dense / 2)
            self.lstm = nn.LSTM(self.params.embed_dense, lstm_num,
                                num_layers, batch_first=True,  # 第一维度是否为batch_size
                                bidirectional=True)  # 双向
        elif params.model_type == 'bigru':
            num_layers = params.rnn_num
            lstm_num = int(self.params.embed_dense / 2)
            self.lstm = nn.GRU(self.params.embed_dense, lstm_num,
                               num_layers, batch_first=True,  # 第一维度是否为batch_size
                               bidirectional=True)  # 双向
        # 全连接分类组件
        self.cls = nn.Linear(params.embed_dense, params.cls_num)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if params.pretrainning_model == 'nezha':
            self.apply(self.init_bert_weights)
        else:
            self.init_weights()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        layer_logits = []
        all_encoder_layers = outputs[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)
        layer_dist = torch.nn.functional.softmax(layer_logits)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)
        word_embed = self.dense_final(pooled_output)
        dym_layer = word_embed
        return dym_layer

    def get_weight_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                cls_label=None):

        # 预训练模型
        if self.params.pretrainning_model == 'nezha':
            encoded_layers, ori_pooled_output = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_all_encoded_layers=True
            )
            sequence_output = encoded_layers[-1]
        else:
            sequence_output, ori_pooled_output, encoded_layers, _ = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # 对预训练模型的改进: 动态权重融合
        if self.params.use_origin_bert == 'dym':
            sequence_output = self.get_dym_layer(encoded_layers)
        elif self.params.use_origin_bert == 'weight':
            sequence_output = self.get_weight_layer(encoded_layers)
            sequence_output = self.dense_final(sequence_output)
        else:
            sequence_output = self.dense_final(sequence_output)  # batch, seq_len, hidden

        # 下游结构
        if self.params.model_type == 'bilstm' or self.params.model_type == 'bigru':
            sequence_output = self.lstm(sequence_output)[0]

        # pooling方式
        if self.params.is_avg_pool == 'max':
            pooled_output = torch.nn.functional.max_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            pooled_output = torch.squeeze(pooled_output, -1)

        elif self.params.is_avg_pool == 'mean':
            pooled_output = torch.nn.functional.avg_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            pooled_output = torch.squeeze(pooled_output, -1)
        elif self.params.is_avg_pool == 'dym':
            maxpooled_output = torch.nn.functional.max_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            maxpooled_output = torch.squeeze(maxpooled_output, -1)
            meanpooled_output = torch.nn.functional.avg_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            meanpooled_output = torch.squeeze(meanpooled_output, -1)
            pooled_output = self.dym_pooling1d(meanpooled_output, maxpooled_output)
        elif self.params.is_avg_pool == 'weight':
            maxpooled_output = torch.nn.functional.max_pool1d(sequence_output.transpose(1,2),self.params.sequence_length)
            maxpooled_output = torch.squeeze(maxpooled_output, -1)
            meanpooled_output = torch.nn.functional.avg_pool1d(sequence_output.transpose(1,2), self.params.sequence_length)
            meanpooled_output = torch.squeeze(meanpooled_output, -1)
            pooled_output = self.weight_pooling1d(meanpooled_output, maxpooled_output)
        else:
            pooled_output = ori_pooled_output
            pooled_output = self.dense_emb_size(pooled_output)

        # 分类
        cls_output = self.dropout(pooled_output)
        classifier_logits = self.cls(cls_output)  # [bacth_size*]

        if cls_label is not None:
            class_loss = nn.CrossEntropyLoss()(classifier_logits, cls_label)
            outputs = class_loss, classifier_logits, encoded_layers  # 后两项为知识蒸馏
        else:
            outputs = classifier_logits, encoded_layers  #

        return outputs

    def dym_pooling1d(self, avpooled_out, maxpooled_out):
        pooled_output = [avpooled_out, maxpooled_out]
        pool_logits = []
        for i, layer in enumerate(pooled_output):
            pool_logits.append(self.dym_pool(layer))
        pool_logits = torch.cat(pool_logits, -1)
        pool_dist = torch.nn.functional.softmax(pool_logits)
        pooled_out = torch.cat([torch.unsqueeze(x, 2) for x in pooled_output], dim=2)
        pooled_out = torch.unsqueeze(pooled_out, 1)
        pool_dist = torch.unsqueeze(pool_dist, 2)
        pool_dist = torch.unsqueeze(pool_dist, 1)
        pooled_output = torch.matmul(pooled_out, pool_dist)
        pooled_output = torch.squeeze(pooled_output)
        return pooled_output
