import os
import math
import time
import random
import warnings
import numpy as np
import scipy.sparse as sp
from itertools import chain
from tqdm import tqdm, trange
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import util
from utils import enumeration
from utils import evaluation
from modules import set2box

EPS = 1e-10

warnings.filterwarnings('ignore')
args = utils.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")

SEED = 2023
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

runtime = time.time()

num_sets, sets, S, M = {}, {}, {}, {}

for dtype in ['train', 'valid', 'test']:
    num_items, num_sets[dtype], sets[dtype] = utils.read_data(args.dataset, args.aspect, args.sim, elems=args.elements, dtype=dtype)
    S[dtype], M[dtype] = utils.incidence_matrix(sets[dtype], num_sets[dtype])

memory = {
    'train': (args.dim * 32 * 2 * num_sets['train']) / 8000,
    'valid': (args.dim * 32 * 2 * num_sets['valid']) / 8000,
    'test': (args.dim * 32 * 2 * num_sets['test']) / 8000
}
    
start_time = time.time()
enumeration = enumeration.Enumeration(sets['train'])
instances, overlaps = enumeration.enumerate_instances(args.pos_instance, args.neg_instance)
instances, overlaps = instances, overlaps
start_time = time.time()
evaluation = evaluation.Evaluation(sets)
start_time = time.time()
model = set2box.model(num_items, args.dim, args.beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

S['train'] = S['train'].to(device)
M['train'] = M['train'].to(device)
overlaps = overlaps.to(device)

for epoch in range(1, args.epochs + 1):
    train_time = time.time()
    
    model.train()
    epoch_loss = 0
    
    batches = utils.generate_batches(len(instances), args.batch_size)
        
    for i in trange(len(batches), position=0, leave=False):

        batch_instances = instances[batches[i]]
        loss = model.forward(S['train'], M['train'], batch_instances, overlaps[batches[i]])
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        model.radius_embedding.weight.data = model.radius_embedding.weight.data.clamp(min=EPS)
        
    train_time = time.time() - train_time
    
    for dtype in ['train', 'valid']:
        pred, _, _ = evaluation.pairwise_similarity(model, S[dtype], M[dtype], args.beta, dtype, True)
        for metric in ['ji', 'di', 'oc', 'cs']:
            mse = mean_squared_error(pred[metric], evaluation.ans[dtype][metric])

aspects = {'S': 'road_segments', 'T': 'timelist', 'ST': 'stroad_segments'}
affix = aspects[args.aspect]
if 'points' in args.elements:
    affix = 'points_' + args.elements.split('_')[1]
if 'hexes' in args.elements:
    affix = 'hexes'
model_name = './model/{}/{}_{}_{}_lr={}_beta={}.pt'.format(args.sim, args.dataset, 'test', affix, args.learning_rate, args.beta)
torch.save(model, model_name)
