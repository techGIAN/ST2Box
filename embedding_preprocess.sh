#!/bin/bash

# note that you can change these settings
python gt_extractor.py --dataset tdrive --sim TP
python traj_emb_preprocess.py -- dataset tdrive --sim TP