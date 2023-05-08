#!/bin/bash

# note that you can change these settings
python utils/gt_extractor.py --dataset tdrive --sim TP
python utils/traj_emb_preprocess.py -- dataset tdrive --sim TP