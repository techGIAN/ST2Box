import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='tdrive', type=str, help='the data')
    parser.add_argument("--sim", default='TP', type=str, help='similarity gt')
    parser.add_argument("--rounder", default=4, type=int, help='for refinement')
    return parser.parse_args()

args = parse_args()

dataset = args.dataset
sim = args.sim
types = ['train', 'valid', 'test', 'query']

# Changeable
counts_dict = {
    'tdrive': [10000, 4000, 4999, 1],
    'nyc': [25000, 8524, 20000, 1]
}
counts = counts_dict[dataset]

offsets = [0, 0, 0, counts[2]]
rounder = args.rounder

unique_set = set()
for typ, ct in zip(types, counts):

    emb_filename = '../traj_embeddings/' + '_'.join([typ, dataset, sim, 'embedding']) + '.npy'
    set_name = '../data/{}/st_traj/{}/{}_stroads_{}.txt'.format(dataset, sim, dataset, typ)

    if typ != 'query':
        emb = np.load(emb_filename, allow_pickle=True)
        emb_norm = (emb-np.min(emb))/(np.max(emb)-np.min(emb))
        emb_round = np.round(emb_norm, rounder)*(10**rounder)
        emb_unique = np.unique(emb_round)
        unique_set = unique_set.union(set(emb_unique))

# create dict
n_items = len(unique_set)
D = dict(zip(unique_set, range(n_items))) 

for typ, ct, os in zip(types, counts, offsets):

    emb_filename = '../traj_embeddings/' + '_'.join([typ, dataset, sim, 'embedding']) + '.npy'
    set_name = '../data/{}/st_traj/{}/{}_stroads_{}.txt'.format(dataset, sim, dataset, typ)

    if typ != 'query':
        emb = np.load(emb_filename, allow_pickle=True)
        emb_norm = (emb-np.min(emb))/(np.max(emb)-np.min(emb))
        emb_round = np.round(emb_norm, rounder)*(10**rounder)

        emb_d = np.vectorize(D.get)(emb_round)
    
    s = '\t'.join([str(n_items), str(ct)]) + '\n'

    for i in range(os, os+ct):
        row = emb_d[i,:]
        list_str = map(str, row)
        s = s + '\t'.join(list_str) + '\n'

    f = open(set_name, 'w')
    f.write(s)
    f.close()
    
