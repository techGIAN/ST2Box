import util
import torch
import torch.nn as nn
import evaluation
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_embeddings', default=False, action='store_true', help='save the embeddings?')
    parser.add_argument('--no_save_embeddings', dest='save_embeddings', action='store_false', help='save the embeddings?')
    parser.add_argument('--load_embeddings', default=False, action='store_true', help='load the embeddings?')
    parser.add_argument('--no_load_embeddings', dest='load_embeddings', action='store_false', help='load the embeddings?')
    parser.add_argument('--save_spatial_embeddings', default=False, action='store_true', help='save the spatial embeddings?')
    parser.add_argument('--no_save_spatial_embeddings', dest='save_spatial_embeddings', action='store_false', help='save the spatial embeddings?')
    parser.add_argument('--load_spatial_embeddings', default=False, action='store_true', help='load the spatial embeddings?')
    parser.add_argument('--no_load_spatial_embeddings', dest='load_spatial_embeddings', action='store_false', help='load the spatial embeddings?')
    parser.add_argument('--save_temporal_embeddings', default=False, action='store_true', help='save the temporal embeddings?')
    parser.add_argument('--no_save_temporal_embeddings', dest='save_temporal_embeddings', action='store_false', help='save the temporal embeddings?')
    parser.add_argument('--load_temporal_embeddings', default=False, action='store_true', help='load the temporal embeddings?')
    parser.add_argument('--no_load_temporal_embeddings', dest='load_temporal_embeddings', action='store_false', help='load the temporal embeddings?')
    parser.add_argument('--no_temporal', default=False, action='store_true', help='no temporal?')
    parser.add_argument('--no_no_temporal', dest='no_temporal', action='store_false', help='no temporal?')
    parser.add_argument('--dataset', default='tdrive', type=str, help='the data')
    parser.add_argument('--lrs', default=0.001, type=float, help='learning rate spatial')
    parser.add_argument('--lrt', default=0.001, type=float, help='learning rate temporal')
    parser.add_argument('--bs', default=1, type=float, help='beta rate spatial')
    parser.add_argument('--bt', default=1, type=float, help='beta rate temporal')
    parser.add_argument('--sim', default='TP', type=str, help='similarity gt')
    parser.add_argument('--elements', default='road_segments', type=str, help='road_segments or points or hexes')
    parser.add_argument('--st2vec', default=False, action='store_true', help='st2vec?')
    parser.add_argument('--no_st2vec', dest='st2vec', action='store_false', help='st2vec')
    return parser.parse_args()

args = parse_args()

suffix = args.elements
if 'points' in args.elements:
    suffix = 'points'
if 'hexes' == args.elements:
    suffix = 'hexes'

# can modify this depending on your need
offsets = {
    'tdrive': 14000,
    'nyc': 33524
}


if args.load_embeddings:
    test_embeddings = torch.load('./box_embeddings/{}/{}_test_st_emb_lrs={}_bs={}_lrt={}_bt={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, args.lrt, args.bt, suffix))
    query_embeddings = torch.load('./box_embeddings/{}/{}_query_st_emb_lrs={}_bs={}_lrt={}_bt={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, args.lrt, args.bt, suffix))

else:

    if args.load_spatial_embeddings:
        spatial_test_embeddings = torch.load('./box_embeddings/{}/{}_test_s_emb_lr={}_beta={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, suffix))
        spatial_query_embeddings = torch.load('./box_embeddings/{}/{}_query_s_emb_lr={}_beta={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, suffix))
    else:
        asp = 'ST' if args.st2vec else 'S'
        elems = 'points_lon' if 'points' in args.elements else args.elements
        elemss = 'points_lat' if elems == 'lon' else None

        elems = 'hexes' if 'hexes' == 'args.elements' else elems

        num_sets, sets, S, M = {}, {}, {}, {}

        for dtype in ['test', 'query']:
            num_items, num_sets[dtype], sets[dtype] = utils.read_data(args.dataset, aspect=asp, sim=args.sim, elems=elems, dtype=dtype)
            S[dtype], M[dtype] = utils.incidence_matrix(sets[dtype], num_sets[dtype])

        evaluation_s = evaluation.Evaluation(sets, gt_needed=False)

        spatial_model_name = './model/{}/{}_{}_{}_lr={}_beta={}.pt'.format(args.sim, args.dataset, 'test', elems, args.lrs, args.bs)
        spatial_model = torch.load(spatial_model_name)

        _, c_test_s, r_test_s = evaluation_s.pairwise_similarity(spatial_model, S['test'], M['test'], args.bs, 'test', True)
        spatial_test_embeddings = torch.cat((c_test_s, r_test_s), -1)

        _, c_query_s, r_query_s = evaluation_s.pairwise_similarity(spatial_model, S['query'], M['query'], args.bs, 'query', True)
        spatial_query_embeddings = torch.cat((c_query_s, r_query_s), -1)
        spatial_query_embeddings = torch.flatten(spatial_query_embeddings)

        if elemss is not None:

            num_sets, sets, S, M = {}, {}, {}, {}

            for dtype in ['test', 'query']:
                num_items, num_sets[dtype], sets[dtype] = utils.read_data(args.dataset, aspect='S', sim=args.sim, elements=elemss, dtype=dtype)
                S[dtype], M[dtype] = utils.incidence_matrix(sets[dtype], num_sets[dtype])

            evaluation_slat = evaluation.Evaluation(sets, gt_needed=False)

            spatial_model_name = './model/{}/{}_{}_{}_lr={}_beta={}.pt'.format(args.sim, args.dataset, 'test', elemss, args.lrs, args.bs)
            spatial_model = torch.load(spatial_model_name)

            _, c_test_s, r_test_s = evaluation_slat.pairwise_similarity(spatial_model, S['test'], M['test'], args.bs, 'test', True)
            spatial_test2_embeddings = torch.cat((c_test_s, r_test_s), -1)

            _, c_query_s, r_query_s = evaluation_s.pairwise_similarity(spatial_model, S['query'], M['query'], args.bs, 'query', True)
            spatial_query2_embeddings = torch.cat((c_query_s, r_query_s), -1)
            spatial_query2_embeddings = torch.flatten(spatial_query2_embeddings)

            spatial_test_embeddings = torch.cat((spatial_test_embeddings, spatial_test2_embeddings), -1)
            spatial_query_embeddings = torch.cat((spatial_query2_embeddings, temporal_query_embeddings), -1)

        if args.save_spatial_embeddings:

            torch.save(spatial_test_embeddings, './box_embeddings/{}/{}_test_s_emb_lr={}_beta={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, suffix))
            torch.save(spatial_query_embeddings, './box_embeddings/{}/{}_query_s_emb_lr={}_beta={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, suffix))

    if (not args.no_temporal) and args.load_temporal_embeddings and suffix != 'stroad_segments' and suffix != 'hexes':
        temporal_test_embeddings = torch.load('./box_embeddings/{}/{}_test_t_emb_lr={}_beta={}.pt'.format(args.sim, args.dataset, args.lrt, args.bt))
        temporal_query_embeddings = torch.load('./box_embeddings/{}/{}_query_t_emb_lr={}_beta={}.pt'.format(args.sim, args.dataset, args.lrt, args.bt))

    elif (not args.no_temporal) and suffix != 'stroad_segments' and suffix != 'hexes':

        num_sets, sets, S, M = {}, {}, {}, {}

        for dtype in ['test', 'query']:
            num_items, num_sets[dtype], sets[dtype] = utils.read_data(args.dataset, aspect='T', sim=args.sim, dtype=dtype)
            S[dtype], M[dtype] = utils.incidence_matrix(sets[dtype], num_sets[dtype])

        evaluation_t = evaluation.Evaluation(sets, gt_needed=False)

        temporal_model_name = './model/{}/{}_{}_{}_lr={}_beta={}.pt'.format(args.sim, args.dataset, 'test', 'timelist', args.lrt, args.bt)
        temporal_model = torch.load(temporal_model_name)

        _, c_test_t, r_test_t = evaluation_t.pairwise_similarity(temporal_model, S['test'], M['test'], args.bt, 'test', True)
        temporal_test_embeddings = torch.cat((c_test_t, r_test_t), -1)

        _, c_query_t, r_query_t = evaluation_t.pairwise_similarity(temporal_model, S['query'], M['query'], args.bt, 'query', True)
        temporal_query_embeddings = torch.cat((c_query_t, r_query_t), -1)
        temporal_query_embeddings = torch.flatten(temporal_query_embeddings)

        if args.save_temporal_embeddings:
            torch.save(temporal_test_embeddings, './box_embeddings/{}/{}_test_t_emb_lr={}_beta={}.pt'.format(args.sim, args.dataset, args.lrt, args.bt))
            torch.save(temporal_query_embeddings, './box_embeddings/{}/{}_query_t_emb_lr={}_beta={}.pt'.format(args.sim, args.dataset, args.lrt, args.bt))


    if (not args.no_temporal) and suffix != 'stroad_segments' and suffix != 'hexes':
        test_embeddings = torch.cat((spatial_test_embeddings, temporal_test_embeddings), -1)
        query_embeddings = torch.cat((spatial_query_embeddings, temporal_query_embeddings), -1)
    else:
        test_embeddings = spatial_test_embeddings
        query_embeddings = spatial_query_embeddings

    if args.save_embeddings:
        torch.save(test_embeddings, './box_embeddings/{}/{}_test_st_emb_lrs={}_bs={}_lrt={}_bt={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, args.lrt, args.bt, suffix))
        torch.save(query_embeddings, './box_embeddings/{}/{}_query_st_emb_lrs={}_bs={}_lrt={}_bt={}_{}.pt'.format(args.sim, args.dataset, args.lrs, args.bs, args.lrt, args.bt, suffix))


sim_dict = dict()
for i in range(test_embeddings.shape[0]):
    sim_dict[i+offsets[args.data]] = torch.dot(test_embeddings[i], query_embeddings)

sim_df = pd.DataFrame({'traj':sim_dict.keys(), 'sim': sim_dict.values()})
sim_df.sort_values(by=['sim'], inplace=True)
sim_df.to_csv('../sim_dfs/{}/{}_{}_sorted_sim_df_lrs={}_bs={}_lrt={}_bt={}_{}.csv'.format(args.sim, args.dataset, 'test', args.lrs, args.bs, args.lrt, args.bt, suffix), index=False)