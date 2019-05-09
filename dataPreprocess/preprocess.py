import pandas as pd
import numpy as np

path = '../data/'
#raw input
inputfile = 'astralscopedom79.fa'
#generated csv
outputfile = 'all79_superfamily.csv'
#class level
CLASSES = ['a','b','c','d','e','f','g','h','i','j','k','l']

def getlabel_class(line):
	ans = line.split(' ')[1].split('.')[0].strip()
	assert(ans in CLASSES)
	return ans

def getlabel_superfamily(line):
	ans = '.'.join(line.split(' ')[1].split('.')[0:3])
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
	labels.append(getlabel_superfamily(l))
	i+=1
	while i<N and not lines[i].startswith('>'):
		tmp_seq+=lines[i].strip()
		i+=1
	seq.append(tmp_seq.lower())

processed=[t[0]+','+t[1]+'\n' for t in zip(labels, seq)]

#write to dir
with open(path+outputfile,'w') as f:
	f.writelines(processed)

#check statistics
stat = pd.Series([len(x) for x in seq])
print(stat.describe())

stat = pd.Series(labels)
print("Number of classes: %d"%len(stat.unique()))

d = stat.value_counts().to_dict()
lst = sorted([d[t] for t in d.keys()])
print("Number of samples: %d"%sum(lst))
print("Number of labels(label freq>=5): %d"%len([x for x in lst if x>=5]))
print("Number of samples(label freq>=5): %d"%sum([x for x in lst if x>=5]))
print("Number of labels(label freq>=10): %d"%len([x for x in lst if x>=10]))
print("Number of samples(label freq>=10): %d"%sum([x for x in lst if x>=10]))

#print(sum([x for x in lst if x>1])/sum(lst))
#print(sum([x for x in lst if x>=5])/sum(lst))
#print(sum([x for x in lst if x>=10])/sum(lst))