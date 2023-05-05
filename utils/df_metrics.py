import numpy as np
import pandas as pd
from scipy.stats import kendalltau as kt


def calc(embedding_set, input_dis_matrix):
    Q = embedding_set[4999,:]
    Simi = []
    c = 0
    for i in range(4999):
        d = np.dot(Q, embedding_set[i])
        Simi.append((d,c))
        c += 1

    Simi = sorted(Simi, key=lambda x:x[0])
    Simi_pos = [x[1] for x in Simi]


    GT_dis = input_dis_matrix[4999,:4999]
    GT_argsort = np.argsort(GT_dis)

    mae = np.mean(np.abs(np.array(Simi_pos)-np.array(GT_argsort)))
    kt_corr, _ = kt(Simi_pos, GT_argsort)

    return mae, kt_corr