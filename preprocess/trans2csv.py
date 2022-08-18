import pandas as pd
from config import Config

config = Config()

def get_text_content(path):
    text_content = []
    label_content = []
    with open(path, 'r') as f:
        data = f.readlines()
        if path == config.base_dir + '训练集/train.csv':
            for d in data:
                l = d.split('\t')
                text_content.append(l[2][:-1])
                label_content.append(l[1])
        if path == config.base_dir + '测试集/test.csv':
            for d in data:
                l = d.split('\t')
                text_content.append(l[1][:-1])
    if label_content:
        return text_content, label_content
    else:
        return text_content

def toDataframe(label_, text_):
    if label_ != None:
        df = pd.DataFrame({
            'label': label_,
            'text': text_
        })
        return df
    else:
        df = pd.DataFrame({
            'text': text_
        })
        return df

if __name__ == '__main__':
    text_, label_= get_text_content(config.base_dir + '训练集/train.csv')
    txt_ = get_text_content(config.base_dir + '测试集/test.csv')
    train_df = toDataframe(label_[1:], text_[1:])
    test_df = toDataframe(label_=None, text_=txt_[1:])
    l = train_df.shape[0]
    train_df[:int(train_df.shape[0]*config.split_rate)].to_csv('train.csv')
    train_df[int(train_df.shape[0]*config.split_rate):].to_csv('dev.csv')
    test_df.to_csv('test.csv')
