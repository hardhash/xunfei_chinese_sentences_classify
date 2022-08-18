# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 14:15
# @Author  : chile
# @Email   : realchilewang@foxmail.com
# @File    : config.py
# @Software: PyCharm


class Config(object):
    def __init__(self):
        self.base_dir = '/home/dale/PycharmProjects/pythonProject/xunfei/main/'  # data path
        self.save_model = self.base_dir + 'Savemodel/'  # 模型路径
        self.result_file = 'result/'
        self.label_list = ['0','1']

        self.warmup_proportion = 0.05
        self.use_bert = True
        self.pretrainning_model = 'nezha'
        self.embed_dense = 512

        self.decay_rate = 0.5  # 学习率衰减参数

        self.train_epoch = 40  # 训练迭代次数

        self.learning_rate = 0.00005  # 下接结构学习率
        self.embed_learning_rate = 5e-5  # 预训练模型学习率

        if self.pretrainning_model == 'roberta':
            model = '/home/dale/models/roberta.base/'  # 中文roberta-base
        elif self.pretrainning_model == 'nezha':
            model = '/home/dale/models/nezha-cn-base/'  # 中文nezha-base
        else:
            raise KeyError('albert nezha roberta bert bert_wwm is need')
        self.cls_num = 2
        self.sequence_length = 128
        self.batch_size = 64

        self.model_path = model

        self.bert_file = model + 'pytorch_model.bin'
        self.bert_config_file = model + 'config.json'
        self.vocab_file = model + 'vocab.txt'

        self.use_origin_bert = 'weight'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'weight':初始化12*1向量
        self.is_avg_pool = 'mean'  # dym, max, mean, cls
        self.model_type = 'bilstm'  # bilstm; bigru

        self.rnn_num = 2
        self.flooding = 0
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        # 模型预测路径
        self.checkpoint_path = "/home/dale/PycharmProjects/pythonProject/xunfei/main/Savemodel/runs_0/1659408843/model_0.8369_0.8369_0.8369_17836.bin"
