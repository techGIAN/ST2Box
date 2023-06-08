import numpy as np
import pandas as pd
import argparse

from scipy.stats import kendalltau as kt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='tdrive', type=str, help='the data')
    parser.add_argument("--lrs", default=0.001, type=float, help='learning rate spatial')
    parser.add_argument("--lrt", default=0.001, type=float, help='learning rate temporal')
    parser.add_argument("--bs", default=1, type=float, help='beta rate spatial')
    parser.add_argument("--bt", default=1, type=float, help='beta rate temporal')
    parser.add_argument("--sim", default='TP', type=str, help='similarity gt')
    parser.add_argument("--elements", default='road_segments', type=str, help='road_segments or points or hexes')
    return parser.parse_args()

args = parse_args()

dataset = args.dataset
simi = args.sim

config = yaml.safe_load(open('../config.yaml'))
k = config['k']

HR, PR, REC, F1 = [], [], [], []

suffix = args.elements
if "points" in args.elements:
    suffix = "points"

gt = np.load('./data/{}/st_traj/{}/{}_gt_test.npy'.format(dataset, simi, dataset))
df = pd.read_csv('./sim_dfs/{}/{}_{}_sorted_sim_df_lrs={}_bs={}_lrt={}_bt={}_{}.csv'.format(simi, dataset, 'test', args.lrs, args.bs, args.lrt, args.bt, suffix))


start, end = 33524,53524 # change accdg to your data
gt_dict = dict(zip(range(start,end), gt[-1,:-1]))
sorted_gt_dict = dict(sorted(gt_dict.items(), key=lambda x:x[1]))

gt_list = list(sorted_gt_dict.keys())[:k]
pred_list = df.traj[:k]

n_gt_list = list(sorted_gt_dict.keys())[k:]
n_pred_list = df.traj[k:]

tp = len(set(gt_list).intersection(set(pred_list)))
tn = len(set(n_gt_list).intersection(set(n_pred_list)))
fp = len(set(pred_list).intersection(set(n_gt_list)))
fn = len(set(gt_list).intersection(set(n_pred_list)))


hr = (tp+tn)/(tp+tn+fp+fn) # also called accuracy
HR.append(hr)

pr = tp/(tp+fp)
PR.append(pr)

rec = tp/(tp+fn)
REC.append(rec)

pr_r = np.inf if pr == 0 else 1/pr
rec_r = np.inf if rec == 0 else 1/rec
f1 = 2/(pr_r + rec_r)
F1.append(f1)

print(f'HR: {HR}')

# MAE
n_queries = 4999 # change this as you want
gt_df = pd.DataFrame({'traj': range(start,end), 'sim': gt[-1,:-1], 'ordinal': range(n_queries)})
gt_df.sort_values(by=['sim'], inplace=True)
df = df[:n_queries]
df = pd.DataFrame({'traj': df.traj, 'sim':df.sim, 'ordinal': range(n_queries)})

gt_arr = np.array(gt_df.ordinal)
pred_arr = np.array(df.ordinal)
array_diff = gt_arr-pred_arr
absolute = np.abs(array_diff)
mae = np.mean(absolute)


print(f'MAE: {mae}')

# KT
kt_corr, _ = kt(gt_arr, pred_arr)
print(f'KT: {kt_corr}')
