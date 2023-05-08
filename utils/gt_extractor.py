import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='tdrive', type=str, help='the data')
    parser.add_argument("--sim", default='TP', type=str, help='similarity gt')
    return parser.parse_args()

args = parse_args()

N = np.load('../data/{}/st_traj/{}/test_st_distance.npy'.format(args.dataset, args.sim), allow_pickle=True)
N = N[:,:5000] # Note that you can change this as you will

np.save('../data/{}/st_traj/{}/{}_gt_test.npy'.format(args.dataset, args.sim, args.dataset), N)
