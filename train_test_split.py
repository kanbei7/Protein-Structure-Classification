import pandas as pd
from sklearn.utils import shuffle
import numpy as np

split_ratio = 0.3
path = './data/'
data = path+'all38_coarse.csv'

df = pd.read_csv(data, header = None)
df.columns = ["label","seq"]

class_lst = []
for c in list(df.label.unique()):
	class_lst.append(df[df.label==c])

train_df = []
test_df = []

for df_c in class_lst:
	tmp = df_c.sample(frac = split_ratio, random_state = np.random.randint(1000)+1)
	test_df.append(tmp)
	train_df.append(df_c.drop(tmp.index))

train_df = pd.concat(train_df)
test_df = pd.concat(test_df)

print(len(train_df))
print(len(test_df))

for i in range(5):
	train_df = shuffle(train_df)
	test_df = shuffle(test_df)

train_df.to_csv(path+'train38.csv', index = False, header = False)
test_df.to_csv(path+'test38.csv', index = False, header = False)
