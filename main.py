import argparse
from fastNLP import embeddings
import fastNLP
from fastNLP.core.callback import LRScheduler
from fastNLP.core.losses import LossInForward
from fastNLP.io.model_io import ModelLoader
import torch
import torch.nn as nn
from fastNLP.embeddings import BertEmbedding,StaticEmbedding,StackEmbedding
from torch.optim import lr_scheduler
from model import *
from embedding import *
from fastNLP.io.loader.conll import CNNERLoader
from fastNLP.io.pipe.conll import _CNNERPipe,MsraNERPipe
from fastNLP import SpanFPreRecMetric
from fastNLP import Trainer
from fastNLP.io import ModelSaver,ModelLoader
from fastNLP import Tester
from fastNLP import LossInForward
from fastNLP import EarlyStopCallback,WarmupCallback,LRScheduler

parser = argparse.ArgumentParser(description='Model parameters')

parser.add_argument('--random_seed', type=int, default=1022, help='Choose random_seed')
parser.add_argument('--embedding', default='bert', help='static | bert')
parser.add_argument('--model', default='BiLSTMCRF',help='Sevral models are available: BiLSTMCRF | Transformer | Linear')
parser.add_argument('--epochs', type=int, default=40, help='Set training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--dropout', type=float, default=0.4,help='Set dropout rate')
parser.add_argument('--lr', type=float, default=5e-3, help='Set learning rate')
parser.add_argument('--optim', default='Adam', help='Choose optimizer')
parser.add_argument('--glyph', type=bool, default=False, help='Whether use glyph embedding')
parser.add_argument('--method', default='sum', help='Glyph Embedding strategy')
parser.add_argument('--fusion', default='concat', help='Fusion strategy : concat | linear | lstm')
parser.add_argument('--pron', type=bool, default=False, help='whether use pronunciation embedding')
parser.add_argument('--dataset', default='resume',help='Choose dataset : resume|sub|weibo|peopledaily|msra')
parser.add_argument('--earlystop', type=bool, default=False, help='Apply EarlyStop')
parser.add_argument('--warmup', type=bool, default=False, help='Apply WarmUp')
parser.add_argument('--lrscheduler', type=bool, default=False, help='Apply LRScheduler')
parser.add_argument('--status', default='train',help='train or test')
parser.add_argument('--device', default='cuda:0',help='Device')

parser.add_argument('--pure_glyph', type=bool, default=False, help='Pure glyph embedding')
parser.add_argument('--pure_pron', type=bool, default=False, help='Pure pron embedding')

args = parser.parse_args()
print("****************Model Settings******************")
if not args.pure_glyph and not args.pure_pron:
    print('use embedding:',args.embedding)
    print('use glyph:',args.glyph)
    print('use pron:',args.pron)
else:
    print('Pure glyph:',args.pure_glyph)
    print('Pure pron:',args.pure_pron)
print('model:',args.model)
print('epochs:',args.epochs)
print('dropout:',args.dropout)
print('learning rate:',args.lr)
print('optimizer:',args.optim)
print('dataset:',args.dataset)


# load dataset first
if args.dataset == 'resume':
    data_bundle = CNNERLoader().load("./dataset/resume")
    data_bundle = _CNNERPipe(encoding_type='bio').process(data_bundle)
    src_vocab = data_bundle.get_vocab('chars')
    tgt_vocab = data_bundle.get_vocab('target')
elif args.dataset == 'sub':
    data_bundle = CNNERLoader().load("./dataset/substitution")
    data_bundle = _CNNERPipe(encoding_type='bio').process(data_bundle)
    src_vocab = data_bundle.get_vocab('chars')
    tgt_vocab = data_bundle.get_vocab('target')
elif args.dataset == 'weibo':
    data_bundle = CNNERLoader().load("./dataset/weibo")
    data_bundle = _CNNERPipe(encoding_type='bio').process(data_bundle)
    src_vocab = data_bundle.get_vocab('chars')
    tgt_vocab = data_bundle.get_vocab('target')
elif args.dataset == 'peopledaily':
    data_bundle = CNNERLoader().load("./dataset/peopledaily")
    data_bundle = _CNNERPipe(encoding_type='bio').process(data_bundle)
    src_vocab = data_bundle.get_vocab('chars')
    tgt_vocab = data_bundle.get_vocab('target')
elif args.dataset == 'msra':
    data_bundle = MsraNERPipe().process_from_file()
    src_vocab = data_bundle.get_vocab('chars')
    tgt_vocab = data_bundle.get_vocab('target')

else:
    raise RuntimeError("Unknown dataset")
    
# print(data_bundle.get_dataset('train')[:4])


if args.embedding == 'bert':
    data_bundle.rename_field('chars', 'words')
    embedding = BertEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='cn-wwm',requires_grad=False,auto_truncate=True)
else:
    embedding = StaticEmbedding(vocab=data_bundle.get_vocab('chars'), model_dir_or_name='cn-char-fastnlp-100d',requires_grad=False)
    data_bundle.rename_field('chars', 'words')


# load glyph embedding
if args.glyph:
    glyph_embedding = Glyph_Embedding(vocab=data_bundle.get_vocab('words'), method=args.method, device=args.device,requires_grad=False)
    if args.fusion != 'lstm':
        embedding = StackEmbedding([embedding,glyph_embedding])
    



# load pron embedding
if args.pron:
    pron_embedding = Pronunciation_Embedding(vocab=data_bundle.get_vocab('words'),device=args.device,requires_grad=False)
    if args.fusion != 'lstm':
        embedding = StackEmbedding([embedding,pron_embedding])


if args.pure_glyph:
    embedding = Glyph_Embedding(vocab=data_bundle.get_vocab('words'), method=args.method, device=args.device,requires_grad=False)
if args.pure_pron:
    embedding = Pronunciation_Embedding(vocab=data_bundle.get_vocab('words'),device=args.device,requires_grad=False)

if args.pure_glyph and args.pure_pron:
    embedding = Glyph_Embedding(vocab=data_bundle.get_vocab('words'), method=args.method, device=args.device,requires_grad=False)
    pron_embedding = Pronunciation_Embedding(vocab=data_bundle.get_vocab('words'),device=args.device,requires_grad=False)
    embedding = StackEmbedding([embedding,pron_embedding])

if args.model == 'BiLSTMCRF':
    if args.fusion == 'concat':
        model = BiLSTMCRF(embed=embedding, fusion=False, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200, dropout=args.dropout, target_vocab=data_bundle.get_vocab('target'))
    elif args.fusion == 'linear':
        model = BiLSTMCRF(embed=embedding, fusion=True, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200, dropout=args.dropout, target_vocab=data_bundle.get_vocab('target'))
    else:
        model = LSTMFUSION(sembed = embedding, gembed = glyph_embedding, pembed = pron_embedding,num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200, dropout=args.dropout, target_vocab=data_bundle.get_vocab('target'))
elif args.model == 'Transformer':
    model = Transformer(embed=embedding, num_classes=len(data_bundle.get_vocab('target')), n_head=8, hidden_dim=400, dropout=args.dropout, target_vocab=data_bundle.get_vocab('target'))
else:
    model = MyLinear(embed=embedding, num_classes=len(data_bundle.get_vocab('target')),target_vocab=data_bundle.get_vocab('target'))

print(model)

# define optimizer
if args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
else:
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)


loss = LossInForward()


callbacks = []

if args.earlystop:
    earlystop = EarlyStopCallback(10)
    callbacks.append(earlystop)

if args.warmup:
    warmup = WarmupCallback()
    callbacks.append(warmup)

if args.lrscheduler:
    lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer,5)
    lrscheduler = LRScheduler(lrscheduler)
    callbacks.append(lrscheduler)

metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'))

if args.status == 'train':
    
    trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer, batch_size=args.batch_size,
                    dev_data=data_bundle.get_dataset('test'), metrics=metric, device=args.device,n_epochs=args.epochs,callbacks=callbacks)
    trainer.train()

    #print(embedding(10))
    saver = ModelSaver("./save/"+args.dataset+'_'+args.model+'_'+str(args.epochs)+'_'+str(args.random_seed)+'_'+str(args.embedding)+'_'+str(args.glyph)+'_'+str(args.pron)+'_'+args.fusion+".pkl")
    saver.save_pytorch(model,False)

    model.eval()
    tester = Tester(data_bundle.get_dataset('dev'), model, metrics=metric, batch_size=args.batch_size, device=args.device)
    tester.test()

else:
    loader = ModelLoader()
    model = loader.load_pytorch_model("./save/"+args.dataset+'_'+args.model+'_'+str(args.epochs)+'_'+str(args.random_seed)+'_'+str(args.embedding)+'_'+str(args.glyph)+'_'+str(args.pron)+'_'+args.fusion+".pkl")
    model.eval()
    tester = Tester(data_bundle.get_dataset('test'), model, metrics=metric, batch_size=args.batch_size, device=args.device)
    tester.test()
    



