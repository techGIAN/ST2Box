#!/bin/bash

# set all your parameters here
DATASET="nyc"
SIM="Frechet"
LRS=0.01            # learning rate for the spatial dimension
BS=1                # beta parameter for the spatial dimension
LRT=0.001           # learning rate for the temporal dimension
BT=1                # beta parameter for the temporal dimension
EPOCHS=100
ELEMENTS1="road_segments"           # your options include points, hexagons, road segments, pathlets
                                    # Note that road segments and pathlets here are spatial; can include temporal dimension by adding spatiotemporal

# ========================================================
# Step 1: Learn the spatiotemporal embeddings
# ========================================================

python main.py --dataset $DATASET --gpu 0 --batch_size 512 --epochs $EPOCHS --learning_rate $LRS --dim $DIM --beta $BS --pos_instance 10 --neg_instance 10 --aspect ST --sim $SIM --elements $ELEMENTS1

# ========================================================
# Step 2: Query the trajectories that are most similar
# ========================================================

python utils/sim_query.py --st2vec --save_spatial_embeddings --save_embeddings --dataset $DATASET --lrs $LRS --lrt $LRT --bs $BS --bt $BT --sim $SIM --elements stroad_segments

# ========================================================
# Step 3: Compute the evaluation metrics.
# ========================================================

python compute_eval_metrics.py --dataset $DATASET --lrs $LRS --lrt $LRT --bs $BS --bt $BT --sim $SIM --elements stroad_segments
