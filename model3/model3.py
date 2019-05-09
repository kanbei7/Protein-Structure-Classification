'''
Model3:
time-batched biLSTM
no stacked layer
for multiclass classification
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirTBLSTM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, n_classes,rnn_hidden_size, batch_size, n_layers = 1, pbatch_size=1, seq_len=500):
		super(BiDirTBLSTM, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.n_classes = n_classes
		self.rnn_hidden_size = rnn_hidden_size
		self.n_layers = n_layers
		self.pbatch_size = pbatch_size
		self.batch_size = batch_size
		self.seq_len = seq_len

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		#TB
		self.tb_size = self.seq_len // self.pbatch_size
		assert(self.tb_size<self.seq_len)
		self.rnn_input_size = self.embedding_dim*self.pbatch_size
		self.rnn = nn.LSTM(self.rnn_input_size, self.rnn_hidden_size, bidirectional = True, num_layers = self.n_layers)

		self.main_clf = nn.Sequential(
			nn.Linear(2*rnn_hidden_size, int(rnn_hidden_size/2)),
			nn.ReLU(),
			nn.Linear(int(rnn_hidden_size/2),int(rnn_hidden_size/2)),
			nn.Linear(int(rnn_hidden_size/2), n_classes)
			)


	def forward(self, seq):
		#[batch_size, seq_length=500, embed_dim]
		embeds = self.word_embeddings(seq)
		#Transformed to [batch_size, seq_length/=pbatch_size, embed_dim*pbatch_size]
		embeds = embeds.view(self.batch_size, self.tb_size, -1)
		_, (rnn_out , _) = self.rnn(embeds.transpose(0,1))
		pred_scores = self.main_clf(torch.cat([rnn_out[0], rnn_out[1]] , dim = 1) )
		return pred_scores 

	def load_embeddings(self, pre_embd):
		self.word_embeddings = nn.Embedding.from_pretrained(pre_embd)

