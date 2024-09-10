'''
This file provides help function
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def five_stroke_2_one_hot(strokes, method='sum'):
    ''' convert a string type (max len 4) to one-hot embedding
    storkes : five storkes
    method : cat -> 100 dims int tensor or sum -> 25 dims int tensor
    '''
    assert isinstance(strokes,str), 'Invalid input'

    if strokes == '':
        if method == 'cat':
            strokes = torch.zeros(100,dtype=int)
        elif method == 'sum':
            strokes = torch.zeros(25,dtype=int)
        else:
            raise RuntimeError('Unknown method!')
    else:
        if method == 'cat':
            strokes = torch.tensor([ord(stroke)-ord('a') for stroke in strokes])

            strokes = F.one_hot(strokes,num_classes =25).view(-1)

            # padding
            while strokes.shape[0] < 100:
                strokes = torch.cat([strokes,torch.zeros(25,dtype=int)],dim=0)
        elif method == 'sum':
            strokes = torch.tensor([ord(stroke)-ord('a') for stroke in strokes])

            strokes = torch.sum(F.one_hot(strokes,num_classes=25),dim=0)
        else:
            raise RuntimeError('Unknown method!')

    return strokes

def pinyin_2_one_hot(pinyin):
    ''' convert pinyin to one-hot embedding -> 27 dims int tensor
    pinyin : a list [aux,vow,tune]([str,str,int])
    '''
    #print(pinyin)

    assert len(pinyin) == 3, 'check the length of pinyin'

    aux_dict = {'b':0,'k':1,'s':2,'t':3,'ʂ':4,'f':5,'ɕ':6,'ʐ':7,'j':8,'n':9,'w':10,'l':11,'m':12,'x':13}
    vow_dict = {'a':0,'e':1,'u':2,'o':3,'i':4,'y':5}
    vow_aux_dict = {'ŋ':0,'n':1}

    # auxiliary
    aux = []
    if pinyin[0] == '':
        aux = torch.zeros(15,dtype=int)
    else:
        for i in range(len(pinyin[0])):
            if i != len(pinyin[0])-1:
                aux.append(aux_dict[pinyin[0][i]])
            else:
                if pinyin[0][i] == '\'':
                    tail = torch.tensor([1])
                else:
                    tail = torch.tensor([0])
                    aux.append(aux_dict[pinyin[0][i]])
                aux = torch.sum(F.one_hot(torch.tensor(aux), num_classes=14),dim=0)
                aux = torch.cat([aux,tail],dim=0)


    # main
    vow = []
    if pinyin[1] == '':
        vow = torch.zeros(8,dtype=int)
    else:
        for i in range(len(pinyin[1])):
            if i != len(pinyin[1])-1:
                vow.append(vow_dict[pinyin[1][i]])
            else:
                if pinyin[1][i] == 'ŋ':
                    tail = torch.tensor([1,0])
                elif pinyin[1][i] == 'n':
                    tail = torch.tensor([0,1])
                else:
                    tail = torch.tensor([0,0])
                    vow.append(vow_dict[pinyin[1][i]])
                vow = torch.sum(F.one_hot(torch.tensor(vow), num_classes=6),dim=0)
                vow = torch.cat([vow,tail],dim=0)

    # tune
    tune = F.one_hot(torch.tensor(pinyin[2]-1),num_classes =4) if pinyin[2]!=0 else torch.zeros(4,dtype=int)

    return torch.cat([aux,vow,tune],dim=0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    '''
    From CSDN 粥老师 https://blog.csdn.net/stupid_3/article/details/83184691
    '''

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        if attn_mask:
        	# 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# 计算softmax
        attention = self.softmax(attention)
		# 添加dropout
        attention = self.dropout(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    '''
    From CSDN 粥老师 https://blog.csdn.net/stupid_3/article/details/83184691
    '''
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
		# multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
		# 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

def padding_mask(seq_k, seq_q):
	# seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

class PositionalEncoding(nn.Module):
    '''
    From CSDN 粥老师 https://blog.csdn.net/stupid_3/article/details/83184691
    '''
    def __init__(self, d_model, max_seq_len):
        """初始化。
        
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
          [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)

class PositionalWiseFeedForward(nn.Module):
    '''
    From CSDN 粥老师 https://blog.csdn.net/stupid_3/article/details/83184691
    '''
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

if __name__ == '__main__':
    print(five_stroke_2_one_hot('vbg',method='sum'),five_stroke_2_one_hot('vbg',method='sum').shape)
    print(pinyin_2_one_hot(['w','ei',1]),pinyin_2_one_hot(['w','ei',1]).shape)
    
