import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from skipgram_model import skipgram
from ngram_model import NGramLanguageModeler
import random

path = './data/'
inputfile = 'all38_coarse.csv'

#window_size = 3
context_size = 4
embedding_dim = 16
batch_size = 10
n_epoch = 80
neg_sample = 10
hidden_size = 32
learning_rate = 1e-4


with open(path+inputfile,'r') as f:
	lines = f.readlines()

seq = ''
raw_data = []
for l in lines:
	seq+=l.split(',')[1].strip()
	raw_data.append(list(l.split(',')[1].strip()))

words = pd.Series(list(seq))
words = sorted(list(words.unique()))
vocab_size = len(words)

words2idx = {}
for i in range(len(words)):
	words2idx[words[i]] = i


#reverse ngram?

ngram_data = []
for k in range(len(raw_data)):
	sentence = raw_data[k]
	tmp_batch = [[words2idx[sentence[i]], words2idx[sentence[i + 1]], words2idx[sentence[i+2]], words2idx[sentence[i + 3]], words2idx[sentence[i + 4]]] for i in range(len(sentence) - context_size)]
	ngram_data += tmp_batch

print(len(ngram_data))
print("data generated")

n_total = len(ngram_data)
#n_sample = int(0.9*n_total)

losses = []
model = NGramLanguageModeler(vocab_size, embedding_dim, context_size, hidden_size, batch_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)


def getbatch(data,batch_size):
	N = int(len(data)/batch_size)
	batches = []
	data = np.array(data)
	for i in range(N):
		batches.append((i, torch.tensor(data[i*batch_size:i*batch_size+batch_size , 0:4], dtype=torch.long),  torch.tensor(data[i*batch_size:i*batch_size+batch_size , 4], dtype=torch.long)))
	return batches, N


for epoch_idx in range(n_epoch):
	total_loss = 0.0
	#sample r=0.4
	#rand_ngram = random.sample(ngram_data, n_sample)
	#print(len(rand_ngram))
	train_loader,n_batches = getbatch(ngram_data, batch_size)
	for batch_idx, context_idxs, target in train_loader:
	#for context, target in rand_ngram:
		#X = Variable(X.transpose(0,1).unsqueeze(2))
		model.zero_grad()
		log_probs = model(context_idxs)
		#print(log_probs.shape)
		loss = F.nll_loss(log_probs, target)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
		#print("%d"%batch_idx)
	losses.append(total_loss)
	print("[Epoch {}] total loss: {:.4f}".format(epoch_idx, total_loss/n_batches))
	if epoch_idx==0:
		model.save_embedding('./ngram/testembed.csv', words)

	if epoch_idx>49:
		model.save_embedding('./ngram/embed16_%d'%epoch_idx+'.csv', words)


print("Finished.")

'''
model = skipgram(self.vocab_size, self.embedding_dim)
optimizer = optim.SGD(model.parameters(),lr=0.03)


def generateNGram(data, ctxt_size, batch_size):
	ngram = []
	for i in range(len(data)):
		ngram
ngram_data, n_batches = generateNGram(raw_data, context_size, batch_size)
'''


