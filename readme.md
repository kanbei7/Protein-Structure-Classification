## CSCI5461 Course Project
### Data
[SCOPe 2.07](http://scop.berkeley.edu/astral/ver=2.07)

### requirements
* python [>=3.6]
* pytoch [>=0.4]
* pandas
* numpy

### Data Preprocessing
* preprocess.py converts ".fa" file to csv file
* dataSelection.py filters the data according to sequence length and the size of each class
* stratifiedKfold_split.py generate equal k splits in all classes and then combines them.

### Train Amino acids embeddings
* some of the best pretrained models are in "emb/"
* cbow and ngram share the same model file
* do onehot by hand or few lines of codes

### Fold-level Structure Classification
* model1 and 2 are to model binary classification
* mdoel3 are made for multiclass classification


