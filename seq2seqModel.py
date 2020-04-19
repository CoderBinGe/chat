# coding:utf-8

import tensorflow as tf
import getConfig

gConfig = {}

gConfig = getConfig.get_config(config_file='seq2seq.ini')


# ---------------------------------------------------------------------------------------------------------------------
# 编码器
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size  # 初始化批处理数据的大小
        self.enc_units = enc_units  # encoder神经元数量
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    # 定义调用函数，进行输入、输出的逻辑变换处理
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    # 初始化隐藏层的神经元
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


# ---------------------------------------------------------------------------------------------------------------------
# attention模型
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # 初始化定义权重网络层W1、W2及最后的打分网络层V，最终打分结果作为注意力的权重值
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # 定义调用函数，进行输入、输出的逻辑变换处理
    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape：(batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # 将score值进行归一化
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        # 将attention_weights的值与输入文本相乘，得到加权过后的文本向量
        context_vector = attention_weights * values
        # 按行求和得到最终的文本向量
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# ---------------------------------------------------------------------------------------------------------------------
# 解码器
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size  # 初始化批处理数据的大小
        self.dec_units = dec_units  # decoder神经元数量
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)  # embedding 层
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 全连接层
        self.fc = tf.keras.layers.Dense(vocab_size)

        # attention
        self.attention = BahdanauAttention(self.dec_units)

    # 定义调用函数，进行输入、输出的逻辑变换处理
    def call(self, x, hidden, enc_output):
        # 解码器输出维度(batch_size, max_length, hidden_size)
        # 计算context_vector和注意力权重
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # 对decoder的输入进行embedding
        x = self.embedding(x)

        # decoder的输入向量与context_vector进行连接
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        # 对输出向量维度变换，变换成(batch_size, vocab)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


# ---------------------------------------------------------------------------------------------------------------------
# 定义整个模型的损失目标函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# 定义损失函数
def loss_function(real, pred):
    # 为了增强训练效果和提高泛化性，将训练数据中常用的词遮罩，构建mask向量
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # 计算损失向量
    loss_ = loss_object(real, pred)
    # 转换mask向量的类型
    mask = tf.cast(mask, dtype=loss_.dtype)
    # 使用mask向量对损失向量进行处理，去除padding引入的噪声
    loss_ *= mask
    # 返回平均损失值
    return tf.reduce_mean(loss_)


# ---------------------------------------------------------------------------------------------------------------------
# 初始化
vocab_inp_size = gConfig['enc_vocab_size']
vocab_tar_size = gConfig['dec_vocab_size']
embedding_dim = gConfig['embedding_dim']
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']

# 实例化
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# 定义Adam优化器
optimizer = tf.keras.optimizers.Adam()

# 实例化Checkpoint类，使用save方法来保存训练模型
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


# ---------------------------------------------------------------------------------------------------------------------
# 定义训练方法，对输入数据进行一次循环训练
def train_step(input, targ, targ_lang, enc_hidden):
    loss = 0
    # 记录梯度求导信息
    with tf.GradientTape() as tape:
        # encoder对输入语句进行编码，得到编码向量enc_output和隐藏层的输出enc_hidden
        enc_output, enc_hidden = encoder(input, enc_hidden)
        dec_hidden = enc_hidden
        # 构建decoder输入向量，首词使用start对应的字典码值作为向量的第一个数值，维度是BATCH_SIZE大小
        dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)

        # 开始训练解码器
        for t in range(1, targ.shape[1]):

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            # 计算loss
            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    # 计算批处理的平均值
    batch_loss = (loss / int(targ.shape[1]))
    # 计算参数变量
    variables = encoder.trainable_variables + decoder.trainable_variables
    # 计算梯度
    gradients = tape.gradient(loss, variables)
    # 优化器优化参数变量的值
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
