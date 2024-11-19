# encoding=utf-8
"""
Author   : Abu
FileName : abu_bert.py
Time     : 2024/8/14
IdeaName  : PyCharm
"""
import math
import numpy as np
import torch
from transformers import BertModel
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
return_dict=False: 当设置为False时，模型的输出是一个元组（tuple）。
这个元组包含多个元素，如模型的最后一层隐藏状态（通常是第一个元素），
以及其他额外的输出（如果有的话，比如注意力权重）

return_dict=True: 当设置为True时，模型的输出是一个字典（dictionary）。
字典中的键包含模型的所有输出名称（例如 last_hidden_state、pooler_output、attentions 等），
并对应于它们的值。这样可以更加直观地访问模型的不同部分输出
"""
bert = BertModel.from_pretrained('./bert-base-chinese', return_dict=False)
state_dict = bert.state_dict()  # 权重值
bert.eval()  # 预测模式
x = np.array([2450, 15486, 102, 2110])  # 理解为4个字的句子
torch_x = torch.LongTensor([x])  # torch形式的输入

seqence_output, pooler_output = bert(torch_x)
"""
hidden_size：可以理解成word_dim

seqence_output：是bert模型最后一层的输出，包含了序列中所有字词的表示向量
【shape(batch_size, sequence_len, hidden_size)】

pooler_output：整句话的表示向量。
具体来说，它是sequence_output中[CLS]标记（即第一个标记）的输出经过一个全连接层（线性层）和 tanh激活函数之后的结果
【shape(batch_size, hidden_size)】

每个词汇的表示都通过自注意力机制与序列中的其他词汇交互。因此，[CLS] 标记的位置最终会包含整个序列的信息，
因为它会与序列中的其他所有标记进行信息交换。
[CLS] 标记在BERT中的作用不仅仅是一个普通的标记符号，而是被设计为整个输入序列的全局表示。
在模型的训练过程中，它的表示逐渐被优化为能够有效地总结整个句子的语义信息，从而在下游任务中作为句子级别的表示
"""


# softmax归一化
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))


class AbuBert:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        # 这里的层数要和预训练config文件中的保持一致，默认是12层
        # 这里只看第一层，文件中的也要是1
        self.num_layers = 1
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # 加载bert中embedding权重
        self.word_embed = state_dict[
            "embeddings.word_embeddings.weight"].numpy()
        self.position_embed = state_dict[
            "embeddings.position_embeddings.weight"].numpy()
        self.segment_embed = state_dict[
            "embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict[
            "embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict[
            "embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        # transformer中编码器层，有多层
        for i in range(self.num_layers):
            # 每个字词的向量会通过query.weight（一个线性变换矩阵）和query.bias（一个偏置向量）来生成对应的Query向量
            # 是用来计算每个字词与其他字词的关系的
            q_w = state_dict[
                f"encoder.layer.{i}.attention.self.query.weight"].numpy()
            q_b = state_dict[
                f"encoder.layer.{i}.attention.self.query.bias"].numpy()
            # 类似q，通过key.weight和key.bias生成Key向量，表示某个字词信息的特征
            k_w = state_dict[
                f"encoder.layer.{i}.attention.self.key.weight"].numpy()
            k_b = state_dict[
                f"encoder.layer.{i}.attention.self.key.bias"].numpy()
            # Value向量表示每个token的实际信息内容,从token的嵌入向量通过线性变换得到的
            v_w = state_dict[
                f"encoder.layer.{i}.attention.self.value.weight"].numpy()
            v_b = state_dict[
                f"encoder.layer.{i}.attention.self.value.bias"].numpy()
            # self-attention机制输出权重和偏置项
            attention_output_weight = state_dict[
                f"encoder.layer.{i}.attention.output.dense.weight"].numpy()
            attention_output_bias = state_dict[
                f"encoder.layer.{i}.attention.output.dense.bias"].numpy()
            # 第一个归一化层权重和偏置项
            attention_layer_norm_w = state_dict[
                f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy()
            attention_layer_norm_b = state_dict[
                f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy()
            # 前馈网络中中间层的全连接层参数，通常用来扩大隐藏层维度
            intermediate_weight = state_dict[
                f"encoder.layer.{i}.intermediate.dense.weight"].numpy()
            intermediate_bias = state_dict[
                f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
            # 前馈网络中输出层的全连接层参数，用于将中间层的输出重新映射到原始的隐藏层维度
            output_weight = state_dict[
                f"encoder.layer.{i}.output.dense.weight"].numpy()
            output_bias = state_dict[
                f"encoder.layer.{i}.output.dense.bias"].numpy()
            # 对前馈网络的输出进行LayerNorm归一化的参数
            ff_layer_norm_w = state_dict[
                f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
            ff_layer_norm_b = state_dict[
                f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
            # 把这些权重名称都保存起来
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight,
                 attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b,
                 intermediate_weight,
                 intermediate_bias,
                 output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        # pooler层，通常用于句子分类任务中，将第一个token（[CLS]）的隐藏层输出转换为句子级别的特征向量
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    # embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    # 按照输入句子x的索引，获取每一行的向量
    def get_embed(self, embed_matrix, x):
        return np.array([embed_matrix[index] for index in x])

    # 归一化层
    def layer_norm(self, x, w, b):
        """
        np.mean(x, axis=1, keepdims=True) 计算输入 x 的均值，均值是沿着 axis=1 计算的，
        这意味着对每个样本的特征维度（通常是隐藏层的所有神经元）计算均值。
        keepdims=True 确保输出的维度与输入 x 相同
        np.std(x, axis=1, keepdims=True) 类似地计算标准差
        然后，将输入 x 减去均值，并除以标准差，从而使 x 的每个样本归一化，得到零均值和单位方差的输出
        """
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1,
                                                             keepdims=True)
        x = x * w + b
        return x

    def embedding_forward(self, x):
        # x.shape = [max_len](bert最大长度为512，如果句子长度大于512，就会截断，所以这里是max_len)
        # 传统embedding层, out shape(max_len, hidden_size[word_dim])
        we = self.get_embed(self.word_embed, x)
        # position embedding的输入 [0, 1, 2, 3]
        # np.array的含义是把句子x中的字转化为下标列表, out shape(max_len, hidden_size[word_dim])
        pe = self.get_embed(self.position_embed, np.array(list(range(len(x)))))
        # segment embedding(token type embedding),用来区分不同句子
        # 第一个句子的 token type 通常全为 0(Ea)，第二个句子的 token type 通常全为 1(Eb)
        # out shape(max_len, hidden_size[word_dim])
        se = self.get_embed(self.segment_embed, np.array([0] * len(x)))
        # 三种embedding加和之后为最后的emdedding层
        embeddding_add = we + pe + se
        # 加和之后，需要过一个归一化层, out shape(max_len, hidden_size[word_dim])
        embedding = self.layer_norm(embeddding_add,
                                    self.embeddings_layer_norm_weight,
                                    self.embeddings_layer_norm_bias)
        return embedding

    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768[word_dim], num_attention_heads=12,attention_head_size=64
        max_len, hidden_size = x.shape
        # 如果x原始形状的总元素数不等于max_len * num_attention_heads * attention_head_size，那么重塑操作将无法执行
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        # swapaxes(axis1, axis2) 是 NumPy 数组的一个方法，它将数组的两个轴交换
        x = x.swapaxes(1, 0)
        return x

    # self-attention的计算
    def self_attention(self, x, qw, qb, kw, kb, vw, vb, attention_output_weight,
                       attention_output_bias, num_attention_head, hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        # q、k、v shape: [max_len, hidden_size], liner = W * X + B
        q = np.dot(x, qw.T) + qb
        k = np.dot(x, kw.T) + kb
        v = np.dot(x, vw.T) + vb
        # 多头机制，按照给定的头数分割hidden_size[word_dim]
        attention_head_size = int(hidden_size / num_attention_head)
        # q、k、v.shape(num_attention_head, max_len, attention_head_size)
        q = self.transpose_for_scores(q, attention_head_size, num_attention_head)
        k = self.transpose_for_scores(k, attention_head_size, num_attention_head)
        v = self.transpose_for_scores(v, attention_head_size, num_attention_head)
        # qk^t / sqrt(dk), qk.shape(num_attention_head, max_len, max_len)
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape =  num_attention_head, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # 多头机制中，需要把多头分割的形状转化为原来的形状，即：(max_len, hidden_size)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    # 前馈网络计算
    def feed_forward(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):
        """
        :param intermediate_weight: intermediate_size, hidden_size
        :param intermediate_bias: intermediate_size
        :param output_weight: hidden_size, intermediate_size
        :param output_bias: hidden_size
        """
        # output shape：[max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x

    def single_transformer_layer_forward(self, x, layer_num_index):
        # 计算单层transformer
        weights = self.transformer_weights[layer_num_index]
        # 取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
            k_w, k_b, \
            v_w, v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            intermediate_weight, intermediate_bias, \
            output_weight, output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = weights
        # self-attention层
        attention_output = self.self_attention(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bias,
                                               self.num_attention_heads, self.hidden_size)

        # bn层，并使用残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)

        # feed forward层
        feed_forward_x = self.feed_forward(x, intermediate_weight, intermediate_bias,
                                           output_weight, output_bias)
        # bn层，并使用残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # 执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    # pooler layer整句话的向量输出
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    # 最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        seqence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(seqence_output[0])
        return seqence_output, pooler_output


# 自制
db = AbuBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)

# torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)
