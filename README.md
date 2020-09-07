# fakenews2020

Text categorization experiments for automatic fake news detection.

## About

This repository contains the Python code for the experiments described in the paper:

> Z. Bodó. Fake news detection without external knowledge. Submitted to MDIS 2020.

## Command-line interface

```
usage: test.py [-h] --dataset DATASET --method METHOD [--text TEXT]
               [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Dataset name.
                        Available datasets:
                        	0 - FNAMT (FakeNewsAMT)
                        	1 - Celeb (Celebrity)
                        	2 - BF (BuzzFeed)
                        	3 - Rand (Random)
                        	4 - MD (McIntire's dataset)
                        	5 - UTK (UTK-MLC)
                        	6 - GC (GossipCop)
                        	7 - PF (PolitiFact)
  --method METHOD, -m METHOD
                        Method.
                        Available methods:
                        	0 - baseline binary
                        	1 - baseline freq
                        	2 - baseline tf-idf
                        	3 - uni+bigrams binary
                        	4 - new tokenizer
                        	5 - chi2 5000
                        	6 - stat. features
                        	7 - tweak1
                        	8 - tweak2
                        	9 - c-cnn
                        	10 - w-cnn
                        	11 - lstm
                        	12 - a-blstm
                        	13 - gcn
  --text TEXT, -t TEXT  Text data to use (default=2).
                        	0 - article
                        	1 - title
                        	2 - title+article
  --gpu GPU, -g GPU     Use GPU if available (default=1 [true]).
```

The datasets were preprocessed as needed for our methods (can be found in the `data` subdirectory), 
but one can find the links to the original corpora in the paper.

The range 0-8 of the methods represents bag-of-words models, while 9-13 are neural network models for text categorization.
For the description of the methods see the paper and/or source code.

For graph convolutional networks the normalized symmetric adjacency matrix (`A`) needs to be calculated and saved.
We uploaded the pre-computed matrices only for the first 4 datasets (because of its size); these are stored in the `gcn` subfolder.
