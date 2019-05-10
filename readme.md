## CSCI5461 Course Project
### Intro
The structure of a protein not only gives clues about its biological functions, but also reveals rich information about its evolution history. Each protein has a structure shaped by its unique amino acid sequence. The key idea in this project is the analogy between sentences in natural languages and amino acid sequences of proteins: sequences as "sentences", amino acids as "words", and biological functions as the semantic meaning of sentences. Thus, it is attempting to apply NLP(Natural Language Processing) techniques. Thanks to the rapid development of genome sequencing techniques, protein sequencing is thriving and many amino acid sequences data are collected. In this study, we obtain protein sequences data and their structure labels from SCOPe database. First, we vectorize proteins by learning vector representations of amino acids using embedding algorithms in NLP. Next, we build a time-batched bidirectional LSTM (Long and Short term Memory) neural network to classify fold-level protein structures. The cross-validated results on   this multiclass classification problem show that the CBOW(Continuous Bag of Words) model borrowed from Word2Vec with a window size of 7 yields best performance.

### Data
[SCOPe 2.07](http://scop.berkeley.edu/astral/ver=2.07)

### Requirements
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


