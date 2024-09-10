'''
This file helps generate glyph and pronunciation embedding
'''
import re
from pypinyin import pinyin
from utils import *
import torch.nn as nn
from fastNLP.embeddings import TokenEmbedding
from fastNLP.core import logger
from fastNLP.core.vocabulary import Vocabulary


class Glyph_Dict():
    raw_file = './embeddings/OK.TXT'

    def __init__(self, dict_path = None):
        if dict_path != None:
            self.raw_file = dict_path
        self.glyph_dict = {}

        self._load_from_files()

    # load from raw_file, rewrite it if your txt file is in different format
    def _load_from_files(self):
        f = open(self.raw_file,'r',encoding='gbk')
        lines = f.readlines()
        f.close()

        for i in range(6,len(lines)):
            lines[i] = re.sub(' +',' ',lines[i])
            temp = lines[i].strip('\n').split(' ')
            self.glyph_dict[temp[0]] = temp[2].strip('\n')

    # look up single character embedding
    def lookup(self,x,method = 'sum'):
        if x in self.glyph_dict.keys():
            return five_stroke_2_one_hot(self.glyph_dict[x],method=method).float()
        else:
            return five_stroke_2_one_hot('',method=method).float()
            


    def len(self):
        return len(self.glyph_dict)

# use pypinyin library
class Pronunciation_Dict():
    raw_file = './embeddings/pinyin.txt'

    def __init__(self, dict_path = None):
        self.left_dict = {'':''}
        self.right_dict = {}

        self._load_from_file()

    def _load_from_file(self):
        f = open(self.raw_file,'r',encoding='utf-8')
        lines = f.readlines()
        f.close()

        for i in range(0,len(lines)):
            if lines[i]!='\n':
                if i > 0 and i <= 24:
                    key = lines[i].split()[0]
                    value = lines[i].split()[1].strip('\n')[1:-1]
                    self.left_dict[key] = value
                elif i > 25:
                    key = lines[i].split()[0]
                    value = lines[i].split()[1].strip('\n')[1:-1]
                    self.right_dict[key] = value

    # look up single character embedding
    def lookup(self,x):
        # judge if Chinese character
        if '\u4e00' <= x <= '\u9fff':
            left = pinyin(x,style=3)[0][0]
            # left part is missed in some situation
            if left == '':
                left = pinyin(x,style=4)[0][0]
                if left == 'y' or left == 'w':
                    pass
                else:
                    left = ''

            # get right part
            right = pinyin(x,style=9)[0][0]
            # tune may be missed when character is '啊'
            tune = right[-1]
            try:
                # print(right)
                tune = int(right[-1])
                right = self.right_dict[right[:-1]]
            except:
                right = self.right_dict[right]
                tune = 1
                
            left = self.left_dict[left]

            return pinyin_2_one_hot([left,right,tune]).float()
        else:
            return pinyin_2_one_hot(['','',0]).float()

class Glyph_Embedding(TokenEmbedding):
    def __init__(self,vocab,requires_grad=True, dropout=0, word_dropout=0, method='sum', device='cpu'):
        super(Glyph_Embedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        self.words_to_words = torch.arange(0,len(vocab)-1)
        self.words_to_words = torch.cat([torch.tensor([0]),self.words_to_words],dim=0).to(device)

        embedding = self.look4embedding(vocab,method)

        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad

    def forward(self, words):
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words

    def look4embedding(self,vocab,method):

        flag = 1
        temp = Glyph_Dict()
        vectors = temp.lookup('',method).view(1,-1)    # create unknown embedding
        for word in vocab:
            char = word[0]

            if vocab.to_index(char)!=vocab.padding_idx and vocab.to_index(char)!=vocab.unknown_idx:
                vectors = torch.cat([vectors,temp.lookup(char,method).view(1,-1)],dim=0)
                flag += 1

        print('Find {} out of {} in Glyph Embedding.'.format(flag,len(vocab)))

        return vectors

    @property
    def weight(self):
        return self.embedding.weight

class Pronunciation_Embedding(TokenEmbedding):
    def __init__(self,vocab,requires_grad=True, dropout=0, word_dropout=0, device='cpu'):
        super(Pronunciation_Embedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        self.words_to_words = torch.arange(0,len(vocab)-1)
        self.words_to_words = torch.cat([torch.tensor([0]),self.words_to_words],dim=0).to(device)

        embedding = self.look4embedding(vocab)

        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        

    def forward(self,words):
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words

    def look4embedding(self,vocab):

        flag = 1
        temp = Pronunciation_Dict()
        vectors = temp.lookup('').view(1,-1)    # create unknown embedding
        for word in vocab:
            char = word[0]

            if vocab.to_index(char)!=vocab.padding_idx and vocab.to_index(char)!=vocab.unknown_idx:
                vectors = torch.cat([vectors,temp.lookup(char).view(1,-1)],dim=0)
                flag += 1

        print('Find {} out of {} in Pron Embedding.'.format(flag,len(vocab)))

        return vectors

    @property
    def weight(self):
        return self.embedding.weight


if __name__ == '__main__':
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io import WeiboNERPipe
    data_bundle = WeiboNERPipe().process_from_file()
    vocab = data_bundle.get_vocab('chars')
    embedding = StaticEmbedding(vocab=vocab, model_dir_or_name='cn-char-fastnlp-100d').cuda()

    #print(embedding.weight)


    print(embedding.words_to_words)

    '''
    #print(vocab.to_index('好'))
    print(embedding(vocab.to_index('好')))

    my = Pronunciation_Embedding(vocab)
    my.cuda()
    

    #print(my.weight)
    #print(my.words_to_words)
    print(my(vocab.to_index('好')))
    '''

    my2 = Glyph_Embedding(vocab)

    print(my2.words_to_words)

    #print(my2.weight)
    #print(my2.words_to_words)
    print(my2(vocab.to_index('好')))

    my3 = Pronunciation_Embedding(vocab)
    print(my3(vocab.to_index('好')).shape)
    
