import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NGramLanguageModeler(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size, hidden_size, batch_size):
		super(NGramLanguageModeler, self).__init__()
		self.batch_size = batch_size
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
		self.linear2 = nn.Linear(hidden_size, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((self.batch_size, 1, -1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=-1)
		return log_probs.squeeze()

	def save_embedding(self, file_name, id2word):
		embeds = self.embeddings.weight.data
		with open(file_name, 'w') as f:
			for idx in range(len(embeds)):
				word = id2word[idx]
				embed = ','.join([str(float(c)) for c in list(embeds[idx].squeeze())])
				f.write(word+','+embed+'\n')