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
training_data = path + 'train38.csv'
testing_data =  path + 'test38.csv'

embedding_dim = 16
rnn_hidden_size = 32
pbatch_size=1
n_classes = 7
n_layers = 1
vocab_size = 22
padding_length = 500

batch_size = 10
n_epoch = 200
learning_rate = 1e-4
clip = 10

criterion = nn.CrossEntropyLoss()
words2idx = {}
label2idx = {}

def load_label2idx():
	global label2idx
	global n_classes
	for i in range(n_classes):
		label2idx[chr(ord('a') + i)]=i

def load_word2idx():
	global path
	global words2idx
	global vocab_size
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
	
	for i in range(len(words)):
		words2idx[words[i]] = (i+1)

#prepare data
def load_data(inputfile, batch_size):
	with open(inputfile,'r') as f:
		lines = f.readlines()
	data = []
	for l in lines:
		tag, seq = l.strip().split(',')
		tmp_acd = [words2idx[x] for x in list(seq)]
		if len(tmp_acd)<=padding_length:
			tmp_acd = [0]*(padding_length-len(tmp_acd)) +tmp_acd
			data.append([label2idx[tag]]+tmp_acd[:padding_length])
		#data.append([label2idx[tag]]+tmp_acd[:padding_length])
	#print(len(data))
	data = np.array(data)
	N_batches = int(data.shape[0]/batch_size)
	batches = []
	for i in range(N_batches):
		batches.append((i, torch.from_numpy(data[i*batch_size:i*batch_size+batch_size, 1:]).long(), torch.from_numpy(data[i*batch_size:i*batch_size+batch_size, 0]).long()))
	return batches

#load embbedings
#load by numpy
def load_embed(inputfile):
	embedding_weights = [[0]*embedding_dim]
	with open(inputfile,'r') as f:
		lines = f.readlines()
	for l in lines:
		embedding_weights.append([float(x) for x in l.strip().split(',')])
	return torch.FloatTensor(np.array(embedding_weights))


def train(epoch_idx):
	total_loss = 0.0
	n_total = 0
	n_correct = 0

	for b_idx, X, y in train_loader:
		optimizer.zero_grad()
		pred_scores = model(X)
		loss = criterion(pred_scores, y)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
		total_loss += loss.item()

		_ , predicted_labels = torch.max(pred_scores, 1)
		n_correct += (predicted_labels == y).sum().item()
		n_total += y.size(0)

	print("[Epoch: {}] Loss:{:.4f}, Acc:{:.4f}".format(epoch_idx, total_loss, n_correct / n_total))



if __name__ == '__main__':
	load_label2idx()
	load_word2idx()
	train_loader = load_data(testing_data,10)
	test_loader = load_data(training_data,10)

	model = BiDirLSTM(vocab_size, embedding_dim, n_classes,rnn_hidden_size, batch_size)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	model.load_embeddings(load_embed(embedding_file))

	for i in range(n_epoch):
		model.train()
		train(i)
		#model.eval()
		#test(i)
