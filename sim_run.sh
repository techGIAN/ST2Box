#!/bin/bash

# Run the similarity modules to generate the ground truths
python gens/spatial_similarity.py
python gens/temporal_similarity.py
python utils/data_utils.py

# Now run the main
python st2vec_train.py