import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import yaml
from df_metrics import calc as dfm

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

def test_model(embedding_set, isvali=False):

    config = yaml.safe_load(open('config.yaml'))
    k = int(config['k'])

    if isvali==True:
        input_dis_matrix = np.load(str(config["path_vali_truth"]))
    else:
        input_dis_matrix = np.load(str(config["path_test_truth"]))

    embedding_set = embedding_set.data.cpu().numpy()

    embedding_dis_matrix = []
    for t in embedding_set:
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb-embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_hr_k = 0

    f_num = 0

    mae, kt = dfm(embedding_set, input_dis_matrix)

    for i in range(len(input_dis_matrix)):
        input_R = np.array(input_dis_matrix[i])
        one_index = []
        for idx, value in enumerate(input_R):
            if value != -1:
                one_index.append(idx)
        input_R = input_R[one_index]
        input_R = input_R[:5000]


        input_k = np.argsort(input_R)[1:k]

        embed_R = np.array(embedding_dis_matrix[i])
        embed_R = embed_R[one_index]
        embed_R = embed_R[:5000]

        embed_k = np.argsort(embed_R)[1:k]

        if len(one_index)>=51:
            f_num += 1
            l_hr_k += len(list(set(input_k).intersection(set(embed_k))))

    hr_k = float(l_recall_k) / (k * f_num)

    return hr_k, mae, kt