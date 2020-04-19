# -*- coding:utf-8 -*-
import os
import sys
import time
import tensorflow as tf
import seq2seqModel
from sklearn.model_selection import train_test_split
import getConfig
import io
import warnings
warnings.filterwarnings('ignore')

gConfig = {}

gConfig = getConfig.get_config(config_file='seq2seq.ini')

vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim = gConfig['embedding_dim']
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']


# 输入、输出语句的最大值（自己定义的）
# max_length_inp, max_length_tar = 20, 20

# 计算最大语句长度
def max_length(tensor):
    return max(len(t) for t in tensor)

# ---------------------------------------------------------------------------------------------------------------------
# 语句处理函数，在所有语句开头结尾加上start和end标识
def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    # print(w)
    return w


# 训练数据预处理函数，在语句的前后加上开始和结束标识
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')  # 去除换行符
    word_pairs = [[preprocess_sentence(w) for w in line.split('\t')] for line in lines[:num_examples]]
    return zip(*word_pairs)


# ---------------------------------------------------------------------------------------------------------------------
# 将语句向量化
def tokenize(lang):
    # num_words:需要保留的最大词数，基于词频(vocab_size)  oov_token:不在词典中的词，一般用“3”表示
    # 生成词典
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=gConfig['enc_vocab_size'], oov_token=3)
    lang_tokenizer.fit_on_texts(lang)
    # 句子转换成单词索引序列
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # 长度补齐
    # tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_inp, padding='post')
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# ---------------------------------------------------------------------------------------------------------------------
# 根据需要按量加载数据
def load_dataset(path, num_examples):
    input_lang, target_lang = create_dataset(path, num_examples)
    input_tensor, input_token = tokenize(input_lang)
    target_tensor, target_token = tokenize(target_lang)
    return input_tensor, input_token, target_tensor, target_token


# 加载数据集
input_tensor, input_token, target_tensor, target_token = load_dataset(gConfig['seq_data'],
                                                                      gConfig['max_train_data_size'])


# 计算训练集中最大语句的长度
max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)


# 训练
# ---------------------------------------------------------------------------------------------------------------------
def train():
    print("Preparing data in %s" % gConfig['train_data'])

    # 划分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
    # 每个epoch需要多少步才能将所有数据训练一遍
    steps_per_epoch = len(input_tensor_train) // gConfig['batch_size']
    print(steps_per_epoch)

    # 模型保存路径
    # checkpoint_dir = gConfig['model_data']
    # ckpt = tf.io.gfile.listdir(checkpoint_dir)
    # if ckpt:
    #     print("reload pretrained model")
    #     # 加载最新的模型文件
    #     seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # 将训练数据随机打乱，防止局部最优解
    BUFFER_SIZE = len(input_tensor_train)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

    # 批量获取数据
    # 在少于batch_size元素的情况下是否应删除最后一批,默认是不删除
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 初始化模型保存路径
    checkpoint_dir = gConfig['model_data']
    # 模型文件保存前缀
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()

    while True:
        # 当前训练开始时间
        start_time_epoch = time.time()
        # 初始化隐藏层
        enc_hidden = seq2seqModel.encoder.initialize_hidden_state()
        total_loss = 0
        # 批量从训练集中取出数据进行训练
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            # 每步训练得到的损失值
            batch_loss = seq2seqModel.train_step(inp, targ, target_token, enc_hidden)
            # 计算一个epoch的综合损失值
            total_loss += batch_loss
            print(batch_loss.numpy())

        # 每步训练消耗的时间
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        # 每步的loss值
        step_loss = total_loss / steps_per_epoch
        # 当前已训练的步数
        current_steps = + steps_per_epoch
        # 当前已训练步数每步的平均耗时
        step_time_total = (time.time() - start_time) / current_steps
        # 每个epoch打印一下训练信息
        print('当前已训练总步数: {} 平均耗时: {}  每步耗时: {} 每步loss {:.4f}'.format(current_steps, step_time_total,
                                                                     step_time_epoch, step_loss.numpy()))

        # 每个epoch保存一下模型文件
        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)
        # 刷新
        sys.stdout.flush()


# ---------------------------------------------------------------------------------------------------------------------
# 加载已保存的模型
def reload_model():
    checkpoint_dir = gConfig['model_data']
    # 加载最新的模型文件
    model = seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    return model


# ---------------------------------------------------------------------------------------------------------------------
# 根据输入来预测下一句的输出
def predict(sentence):
    # checkpoint_dir = gConfig['model_data']
    # # 加载最新的模型文件
    # seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # 对输入语句进行处理，开头结尾加上标识符
    sentence = preprocess_sentence(sentence)
    # 向量化转换
    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]
    # 按最大长度补齐
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    # 输入语句转化为tensor
    inputs = tf.convert_to_tensor(inputs)

    # 预测结果
    result = ''
    # 初始化隐藏层
    hidden = [tf.zeros((1, units))]
    # 对输入向量编码
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)

    # 加载已训练好的模型
    model = reload_model()

    # 初始化decoder的隐藏层
    dec_hidden = enc_hidden
    # 初始化decoder的输入
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)

    # 按照语句的最大长度预测输出语句
    for t in range(max_length_tar):
        # predictions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input, dec_hidden, enc_out)
        predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_out)
        # 获取预测结果中向量最大值的index
        predicted_id = tf.argmax(predictions[0]).numpy()

        # 如果预测的结果是结束标识，停止预测
        if target_token.index_word[predicted_id] == 'end':
            break
        result += target_token.index_word[predicted_id] + ' '

        # 将预测的数值作为上文输入信息加入decoder中来预测下一个值
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = getConfig.get_config(sys.argv[1])
    else:

        gConfig = getConfig.get_config()

    # 打印当前执行器的模式
    print('\n>> Mode : %s\n' % (gConfig['mode']))

    if gConfig['mode'] == 'train':  # 训练模式
        train()
    elif gConfig['mode'] == 'serve':  # 服务模式

        print('Serve Usage : >> python3 app.py')
