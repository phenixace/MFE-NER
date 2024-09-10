import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.core.const import Const as C
from fastNLP.modules.encoder import LSTM
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.modules.decoder import ConditionalRandomField
from fastNLP.modules.decoder.crf import allowed_transitions
from torch.nn import LayerNorm
from utils import MultiHeadAttention,padding_mask,sequence_mask,PositionalWiseFeedForward

class BiLSTMCRF(nn.Module):
    def __init__(self, embed, fusion, num_classes, num_layers=1, hidden_size=200, dropout=0.5, target_vocab=None):
        super(BiLSTMCRF,self).__init__()
        self.embed = get_embeddings(embed)

        self.layer_norm = LayerNorm(self.embed.embedding_dim)
        if fusion:
            self.dense = nn.Linear(self.embed.embedding_dim,self.embed.embedding_dim)
        self.fusion = fusion
        if num_layers>1:
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True, dropout=dropout)
        else:
            self.lstm = LSTM(self.embed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab)==num_classes, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, words, seq_len=None, target=None):
        vectors = self.embed(words)

        vectors = self.layer_norm(vectors)
        if self.fusion:
            vector = self.dense(vectors)

        feats, _ = self.lstm(vectors, seq_len=seq_len)
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {C.OUTPUT:pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {C.LOSS:loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)

class LSTMFUSION(nn.Module):
    def __init__(self, sembed, gembed, pembed, num_classes, num_layers=1, hidden_size=200, dropout=0.5, target_vocab=None):
        super(LSTMFUSION,self).__init__()
        self.sembed = get_embeddings(sembed)
        self.gembed = get_embeddings(gembed)
        self.pembed = get_embeddings(pembed)

        self.layer_norm_s = LayerNorm(self.sembed.embedding_dim)
        self.layer_norm_g = LayerNorm(self.gembed.embedding_dim)
        self.layer_norm_p = LayerNorm(self.pembed.embedding_dim)

        if num_layers>1:
            self.lstm_s = LSTM(self.sembed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True, dropout=dropout)
            self.lstm_g = LSTM(self.gembed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True, dropout=dropout)
            self.lstm_p = LSTM(self.pembed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True, dropout=dropout)
        else:
            self.lstm_s = LSTM(self.sembed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True)
            self.lstm_g = LSTM(self.gembed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True)
            self.lstm_p = LSTM(self.pembed.embedding_dim, num_layers=num_layers, hidden_size=hidden_size, bidirectional=True,
                             batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc_s = nn.Linear(hidden_size*2, num_classes)
        self.fc_g = nn.Linear(hidden_size*2, num_classes)
        self.fc_p = nn.Linear(hidden_size*2, num_classes)

        self.fc = nn.Linear(3*num_classes,num_classes)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab)==num_classes, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, words, seq_len=None, target=None):
        vectors = self.sembed(words)
        vectorg = self.gembed(words)
        vectorp = self.pembed(words)

        vectors = self.layer_norm_s(vectors)
        vectorg = self.layer_norm_g(vectorg)
        vectorp = self.layer_norm_p(vectorp)

        feats, _ = self.lstm_s(vectors, seq_len=seq_len)
        featg, _ = self.lstm_g(vectorg, seq_len=seq_len)
        featp, _ = self.lstm_p(vectorp, seq_len=seq_len)

        feats = self.fc_s(feats)
        featg = self.fc_g(featg)
        featp = self.fc_p(featp)

        feats = self.dropout(feats)
        featg = self.dropout(featg)
        featp = self.dropout(featp)
        feat = torch.cat([feats,featg,featp],2)
 
        feat = self.fc(feat)
        assert(feat.shape == feats.shape)

        logits = F.log_softmax(feat, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {C.OUTPUT:pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {C.LOSS:loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)

class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output

class Transformer(nn.Module):
    def __init__(self, embed, num_classes, n_head=8, hidden_dim=1024, dropout=0.5, target_vocab=None):
        super(Transformer,self).__init__()
        self.embed = get_embeddings(embed)

        self.layer_norm = LayerNorm(self.embed.embedding_dim)

        self.transformer = EncoderLayer(model_dim = self.embed.embedding_dim, num_heads = n_head, ffn_dim = hidden_dim, dropout = dropout)

        self.linear = nn.Linear(self.embed.embedding_dim,num_classes)
        self.dropout = nn.Dropout(dropout)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab)==num_classes, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)


    def _forward(self, words, seq_len=None, target=None):
        vectors = self.embed(words)

        vectors = self.layer_norm(vectors)

        feats = self.transformer(vectors)

        feats = self.linear(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {C.OUTPUT:pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {C.LOSS:loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)

class MyLinear(nn.Module):
    def __init__(self, embed, num_classes, dropout=0.5, target_vocab=None):
        super(MyLinear,self).__init__()

        self.embed = get_embeddings(embed)

        self.layer_norm = LayerNorm(self.embed.embedding_dim)

        self.linear = nn.Linear(self.embed.embedding_dim,num_classes)
        self.dropout = nn.Dropout(dropout)

        trans = None
        if target_vocab is not None:
            assert len(target_vocab)==num_classes, "The number of classes should be same with the length of target vocabulary."
            trans = allowed_transitions(target_vocab.idx2word, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, words, seq_len=None, target=None):
        vectors = self.embed(words)

        vectors = self.layer_norm(vectors)

        feats = self.linear(vectors)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)

        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {C.OUTPUT:pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {C.LOSS:loss}

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)
