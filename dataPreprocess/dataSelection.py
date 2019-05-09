import pandas as pd
#filter data

def getfold(x):
	return '.'.join(x.split('.')[:2])

NUMBERofSEQ_PER_FOLD = 30
MAX_LEN = 500
data_path = '../data/'
df = pd.read_csv(data_path + "all79_superfamily.csv", header = None)
df.columns = ["splabel","seq"]
df["foldlabel"] = df.splabel.apply(getfold)
df["seqlen"] = df.seq.apply(lambda x:len(list(x)))
df = df[df.seqlen<(MAX_LEN+1)]

fold_d = df.foldlabel.value_counts().to_dict()
lst = sorted([fold_d[t] for t in fold_d.keys()], reverse = True)
selected_fold = [t for t in fold_d.keys() if fold_d[t]>(NUMBERofSEQ_PER_FOLD-1)]
df = df[df.foldlabel.isin(selected_fold)]

lst = sorted([(t, fold_d[t]) for t in fold_d.keys()], key = lambda x:x[1],reverse = True)
print(lst[:20])

#print("%.4f"%(max([fold_d[t] for t in fold_d.keys() if fold_d[t]>(NUMBERofSEQ_PER_FOLD-1)]) / len(df)))

df = df[['foldlabel','splabel','seqlen','seq']]
df.to_csv(data_path+"selected79_fold_sp.csv")

#print("Number of proteins: %d"%len(df))
#print("Number of folds: %d"%len(df.foldlabel.unique()))
#print("Protein length stats:")
#print(df.seqlen.describe())

