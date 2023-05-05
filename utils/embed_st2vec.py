import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import yaml


def compute_embedding(net, road_network, test_traj, test_time, test_batch):

    if len(test_traj) <= test_batch:
        embedding = net(road_network, test_traj, test_time)
        return embedding
    else:
        i = 0
        all_embedding = []
        while i < len(test_traj):
            embedding = net(road_network, test_traj[i:i+test_batch], test_time[i:i+test_batch])
            all_embedding.append(embedding)
            i += test_batch

        all_embedding = torch.cat(all_embedding,0)
        return all_embedding

def embed_model(embedding_set, typ, isvali=False):

    embedding_set = embedding_set.data.cpu().numpy()
    np.save('./embeds/{}_nyc_TP_embedding.npy'.format(typ), embedding_set)
    return

    