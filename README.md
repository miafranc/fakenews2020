# fakenews2020

Text categorization experiments for automatic fake news detection.

## About

This repository contains the Python code for the tests as described in the paper:

> Z. Bodó. Fake news detection using no external knowledge. Submitted to MDIS 2020.

## Command-line interface

```
usage: test.py [-h] --dataset DATASET --method METHOD [--text TEXT]

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
                        	0 - baseline_binary
                        	1 - baseline_freq
                        	2 - baseline_tfidf
                        	3 - uni+bi_binary
                        	4 - new_tokenizer
                        	5 - chi2_5000
                        	6 - stat_features
                        	7 - tweak1
                        	8 - tweak2
                        	9 - cnn_char
                        	10 - cnn_word
                        	11 - lstm
                        	12 - blstm
                        	13 - gcn
  --text TEXT, -t TEXT  Text data to use (default=2).
                        	0 - article
                        	1 - title
                        	2 - title+article

```

The datasets were preprocessed as needed for our methods (can be found in the `data` subdirectory), 
but one can find the links to the original corpora in the paper.

The range 0-8 of the methods represents bag-of-words models, while 9-13 are neural network models for text categorization.
For the description of the methods see the paper and/or source code.
