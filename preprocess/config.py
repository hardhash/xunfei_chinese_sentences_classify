class Config(object):
    def __init__(self):
        self.base_dir = '/home/dale/PycharmProjects/pythonProject/xunfei/'  # 原始数据路径  trans2csv.py
        self.data_dir = '/home/dale/PycharmProjects/pythonProject/xunfei/preprocess/' # 数据重写后保存读取路径 preprocess.py
        self.save_dir = '/home/dale/PycharmProjects/pythonProject/xunfei/main/'  # 数据规范化后最终的存放路径  preprocess.py
        self.split_rate = 0.9