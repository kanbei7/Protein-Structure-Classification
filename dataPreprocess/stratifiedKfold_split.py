import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from math import ceil
path = '../data/'
data = path+'selected79_fold_sp.csv'
kfold = 5
cv5 = [[] for i in range(kfold)]

df = pd.read_csv(data)
df = df[["foldlabel","splabel","seq"]]
class_lst = []
for c in list(df.foldlabel.unique()):
	class_lst.append(df[df.foldlabel==c])

#For each classs, split into k folds
for df_c in class_lst:
	shuffled_idx = np.random.permutation(df_c.index) 
	fold_size = len(df_c) / kfold 
	for j in range(kfold):
		tmpidx = shuffled_idx[ceil( j*fold_size):ceil((j+1)*fold_size)]
		cv5[j].append(df.iloc[tmpidx])

#concat and get k data sets
for j in range(kfold):
	cv5[j] = pd.concat(cv5[j])

#shuffle data
for i in range(8):
	for j in range(kfold):
		cv5[j] = shuffle(cv5[j])

#write each fold
for j in range(kfold):
	cv5[j].to_csv(path+"s79fold_k%d.csv"%j, index = False, header = False)