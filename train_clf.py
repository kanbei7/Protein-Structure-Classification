import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from random import shuffle
from model_clf import BiDirLSTM

path = './data/'
embedding_file = './ngram/embed16_60.csv'
training_data =
testing_data =  

embedding_dim = 16
rnn_hidden_size = 32
pbatch_size=1
n_classes = 7
n_layers = 1

batch_size = 10
n_epoch = 200
learnig_rate = 1e-4
padding_length = 400
criterion = nn.CrossEntropyLoss()

label2idx = {}
for i in range(n_classes):
	label2idx[chr(ord('a') + i)]=i

with open(path+'all38_coarse.csv','r') as f:
	lines = f.readlines()
seq = ''
raw_data = []
for l in lines:
	seq+=l.split(',')[1].strip()
	raw_data.append(list(l.split(',')[1].strip()))

words = pd.Series(list(seq))
words = sorted(list(words.unique()))
vocab_size = len(words)+1

words2idx = {}
for i in range(len(words)):
	words2idx[words[i]] = (i+1)

#prepare data
def load_data(inputfile, batch_size):
	with open(path+inputfile,'r') as f:
		lines = f.readlines()
	data = []
	for l in lines:
		tag, seq = l.strip().split(',')
		tmp_acd = [words2idx[x] for x in list(seq)]
		if len(tmp_acd<padding_length):
			tmp_acd = [0]*(padding_length-len(tmp_acd)) +tmp_acd
		data.append([label2idx[tag]]+tmp_acd[:padding_length])
	return np.array(data)

#load embbedings
#load by numpy
#flexible path
def load_embbedings(inputfile):
	embedding_weights = [[0.0]*embedding_dim]
	with open(path+inputfile,'r') as f:
		lines = f.readlines()
	for l in lines:
		embedding_weights.append([float(x) for x in l.strip().split(',')])
	return np.array(embedding_weights)

