import pandas as pd
import numpy as np

path = './data/'
inputfile = 'astralscopedom38.fa'
outputfile = 'all38_coarse.csv'
CLASSES = ['a','b','c','d','e','f','g','h','i','j','k','l']

def getlabel(line):
	ans = line.split(' ')[1].split('.')[0].strip()
	assert(ans in CLASSES)
	return ans

with open(path+inputfile,'r') as f:
	lines = f.readlines()

N=len(lines)
i=0
tmp_seq = ''
seq = []
labels = []
while i<N:
	l = lines[i]
	tmp_seq = ''
	labels.append(getlabel(l))
	i+=1
	while i<N and not lines[i].startswith('>'):
		tmp_seq+=lines[i].strip()
		i+=1
	seq.append(tmp_seq.lower())

processed=[t[0]+','+t[1]+'\n' for t in zip(labels, seq)]
with open(path+outputfile,'w') as f:
	f.writelines(processed)

#stat = pd.Series([len(x) for x in seq])
#print(stat.describe())

#stat = pd.Series(labels)
#print(stat.value_counts())