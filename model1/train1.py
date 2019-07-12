'''
Train model1:
biLSTM without time-batched

Task:
Binary classification

Evaluation:
k-fold cross validated score for:
- PR-AUC

'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from random import shuffle
from model1 import BiDirLSTM
from sklearn.metrics import average_precision_score
import time

path = '../data/'
#set to the selected embeddings
embedding_file = '../emb/cbowembed16_w3_50.csv'
data_prefix =  path+'s79fold_k'
#Construct binary labels
#the target label will be marked as 1, other classes as 0
target_label = 'a.1'
#specify the test fold
test_fold = 1

#for ngram and cbow
embedding_dim = 16
#for onehot should be:
#embedding_dim = 23

rnn_hidden_size = 40
pbatch_size= 1
n_classes = 2
n_layers = 1
vocab_size = 24
padding_length = 500

batch_size = 10
n_epoch = 200
learning_rate = 3e-4
clip = 15

criterion = nn.CrossEntropyLoss()
words2idx = {}

def load_word2idx():
	global path
	global words2idx
	global vocab_size
	with open(path+'all79_superfamily.csv','r') as f:
		lines = f.readlines()
	seq = ''
	raw_data = []
	for l in lines:
		seq+=l.split(',')[1].strip()
		raw_data.append(list(l.split(',')[1].strip()))

	words = pd.Series(list(seq))
	words = sorted(list(words.unique()))
	
	for i in range(len(words)):
		words2idx[words[i]] = (i+1)

#Construct binary labels
def getLabel(s):
	return int(s==target_label)

#prepare data
def load_data(fileprefix, test_k,batch_size):
	with open(fileprefix + str(test_k) +'.csv','r') as f:
		lines = f.readlines()
	data = []
	for l in lines:
		tag,_,seq = l.strip().split(',')
		tmp_acd = [words2idx[x] for x in list(seq)]
		if len(tmp_acd)<=padding_length:
			tmp_acd = [0]*(padding_length-len(tmp_acd)) +tmp_acd
			data.append([getLabel(tag)]+tmp_acd[:padding_length])
		#data.append([label2idx[tag]]+tmp_acd[:padding_length])
	#print(len(data))
	data = np.array(data)
	N_batches = int(data.shape[0]/batch_size)
	test_batches = []
	for i in range(N_batches):
		test_batches.append((i, torch.from_numpy(data[i*batch_size:i*batch_size+batch_size, 1:]).long(), torch.from_numpy(data[i*batch_size:i*batch_size+batch_size, 0]).long()))
	
	lines = []
	for i in range(5):
		if i==test_k:
			continue
		with open(fileprefix + str(i) +'.csv','r') as f:
			lines += f.readlines()
	data = []
	for l in lines:
		tag,_,seq = l.strip().split(',')
		tmp_acd = [words2idx[x] for x in list(seq)]
		if len(tmp_acd)<=padding_length:
			tmp_acd = [0]*(padding_length-len(tmp_acd)) +tmp_acd
			data.append([getLabel(tag)]+tmp_acd[:padding_length])	
	data = np.array(data)
	N_batches = int(data.shape[0]/batch_size)
	train_batches = []
	for i in range(N_batches):
		train_batches.append((i, torch.from_numpy(data[i*batch_size:i*batch_size+batch_size, 1:]).long(), torch.from_numpy(data[i*batch_size:i*batch_size+batch_size, 0]).long()))
	
	assert(len(train_batches)>len(test_batches))
	return train_batches, test_batches

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
	y_test = []
	y_score = []
	for b_idx, X, y in train_loader:
		optimizer.zero_grad()
		pred_scores = model(X)
		y_test+=(y.tolist())
		y_score+= pred_scores[:,1].tolist()
		loss = criterion(pred_scores, y)
		loss.backward()
		#nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
		total_loss += loss.item()

		_ , predicted_labels = torch.max(pred_scores, 1)
		n_correct += (predicted_labels == y).sum().item()
		n_total += y.size(0)
	auc_pr = average_precision_score(y_test, y_score)
	print("[Epoch: {}] Loss:{:.4f}, Acc:{:.4f}, AUCPR:{:.4f}".format(epoch_idx, total_loss, n_correct / n_total, auc_pr))

def test(epoch_idx):
	total_loss = 0.0
	n_total = 0
	n_correct = 0
	y_test = []
	y_score = []
	with torch.no_grad():
		for b_idx, X, y in test_loader:
			pred_scores = model(X)
			y_test+=(y.tolist())
			y_score+= pred_scores[:,1].tolist()
			loss = criterion(pred_scores, y)
			total_loss += loss.item()

			_ , predicted_labels = torch.max(pred_scores, 1)
			n_correct += (predicted_labels == y).sum().item()
			n_total += y.size(0)
	auc_pr = average_precision_score(y_test, y_score)
	print("[Test: {}] Loss:{:.4f}, Acc:{:.4f}, AUCPR:{:.4f}".format(epoch_idx, total_loss, n_correct / n_total, auc_pr))

if __name__ == '__main__':
	load_word2idx()
	train_loader, test_loader = load_data(data_prefix, test_fold,10)
	print("Data Loaded.")

	model = BiDirLSTM(vocab_size, embedding_dim, n_classes,rnn_hidden_size, batch_size)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	model.load_embeddings(load_embed(embedding_file))

	for i in range(n_epoch):
		start = time.time()
		model.train()
		train(i)
		model.eval()
		test(i)
		print("Running time: %.2f"%(time.time()-start))
