import pandas as pd
import re
from config import Config

config = Config()

test = pd.read_csv(config.data_dir + 'test.csv', encoding='utf8')
train_df = pd.read_csv(config.data_dir + 'train.csv', encoding='utf8')
dev_df = pd.read_csv(config.data_dir + 'dev.csv', encoding='utf8')

def cal_text_len(row):
    row_len = len(row)
    if row_len < 256:
        return 256
    elif row_len < 384:
        return 384
    elif row_len < 512:
        return 512
    else:
        return 1024

def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('{IMG:.?.?.?}', '', x)
    x = re.sub('<!--IMG_\d+-->', '', x)
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)  # 过滤网址
    x = re.sub('<a[^>]*>', '', x).replace("</a>", "")  # 过滤a标签
    x = re.sub('<P[^>]*>', '', x).replace("</P>", "")  # 过滤P标签
    x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")  # 过滤strong标签
    x = re.sub('<br>', ',', x)  # 过滤br标签
    # 过滤www开头的网址
    x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")
    x = re.sub('\s', '', x)  # 过滤不可见字符
    x = re.sub('Ⅴ', 'V', x)

    # 删除奇怪标点
    for wbad in additional_chars:
        x = x.replace(wbad, '')
    return x

def testProprecess(path):
    df = pd.read_csv(path)
    df['label'] = [0] * df.shape[0]
    return df

train_df['text_len'] = train_df['text'].apply(cal_text_len)
dev_df['text_len'] = dev_df['text'].apply(cal_text_len)
print(train_df['text_len'].value_counts())
print(dev_df['text_len'].value_counts())
print('-------------------')

additional_chars = set()
for t in list(train_df['text']):
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))
print('文中出现的非中英文的数字符号：', additional_chars)
# 一些需要保留的符号
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
print('保留的标点:', extra_chars)
additional_chars = additional_chars.difference(extra_chars)

train_df['text'].apply(stop_words)
dev_df['text'].apply(stop_words)
test['text'].apply(stop_words)

train_df[['text', 'label']].to_csv(config.save_dir + 'train.csv', encoding='utf-8')
dev_df[['text', 'label']].to_csv(config.save_dir + 'dev.csv', encoding='utf-8')
test['text'].to_csv(config.save_dir + 'test.csv', encoding='utf-8', index=False)
testProprecess('test.csv').to_csv(config.save_dir + 'test.csv', encoding='utf-8', index=False)

