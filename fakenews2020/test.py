import sys
import argparse

from utils import load_data
from goftc import GOFTC
from nntc import NNTC
from gcntc import GCNTC

texts = [
    'article', #0
    'title', #1
    'title+article' #2
]

methods = [
    'baseline_binary', #0
    'baseline_freq', #1
    'baseline_tfidf', #2
    'uni+bi_binary', #3
    'new_tokenizer', #4
    'chi2_5000', #5
    'stat_features', #6
    'tweak1', #7
    'tweak2', #8
    'cnn_char', #9
    'cnn_word', #10
    'lstm', #11,
    'blstm', #12
    'gcn' #13
]

datasets = [
    # 0
    {'name': 'FakeNewsAMT', 
     'abbrv': 'FNAMT',
     'path': 'data/fnamt.json',
     'length_chars': 700,
     'length_words': 150},
    # 1
    {'name': 'Celebrity', 
     'abbrv': 'Celeb',
     'path': 'data/celeb.json',
     'length_chars': 3000,
     'length_words': 500},
    # 2
    {'name': 'BuzzFeed', 
     'abbrv': 'BF',
     'path': 'data/bf.json',
     'length_chars': 6000,
     'length_words': 1100},
    # 3
    {'name': 'Random', 
     'abbrv': 'Rand',
     'path': 'data/rand.json',
     'length_chars': 4000,
     'length_words': 700},
    # 4
    {'name': 'McIntire\'s dataset', 
     'abbrv': 'MD',
     'path': 'data/md.json',
     'length_chars': 5000,
     'length_words': 1000},
    # 5
    {'name': 'UTK-MLC', 
     'abbrv': 'UTK',
     'path': 'data/utk.json',
     'length_chars': 5000,
     'length_words': 900},
    # 6
    {'name': 'GossipCop', 
     'abbrv': 'GC',
     'path': 'data/gc.json',
     'length_chars': 4000,
     'length_words': 700},
    # 7
    {'name': 'PolitiFact', 
     'abbrv': 'PF',
     'path': 'data/pf.json',
     'length_chars': 8000,
     'length_words': 1700}
]

def main(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', '-d', type=int, required=True, help="Dataset name.\nAvailable datasets:\n" + "".join([f"\t{i} - {datasets[i]['abbrv']} ({datasets[i]['name']})\n" for i in range(len(datasets))]))
    parser.add_argument('--method', '-m', type=int, required=True, help="Method.\nAvailable methods:\n" + "".join([f"\t{i} - {methods[i]}\n" for i in range(len(methods))]))
    parser.add_argument('--text', '-t', type=int, default=2, help="Text data to use (default=2).\n" + "".join([f"\t{i} - {texts[i]}\n" for i in range(len(texts))]))
    
    args = parser.parse_args(args)

#     print(args)

    data = load_data(datasets[args.dataset]['path'])
    data_prop = datasets[args.dataset]
    
    if args.method in range(0, 9):  # GOFTC
        experiment = GOFTC()
    elif args.method in range(9, 13):  # NN models
        experiment = NNTC()
    elif args.method == 13:  # GCN
        experiment = GCNTC()
    
    print(f"-> Dataset: {data_prop['abbrv']} ({data_prop['name']})\n-> Method: {methods[args.method]}\n-> Text: {texts[args.text]}")
    
    experiment.set_data(data, data_prop)
    experiment.run(args.text, args.method)
    

if __name__ == "__main__":
#     main(sys.argv[1:])
#     main('-d 0 -m 1'.split())
#     main('-d 3 -m 8'.split())
    main('-d 3 -m 12'.split())
#     main('-d 0 -m 13'.split())
#     main('-h'.split())
    