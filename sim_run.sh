#!/bin/bash

# Run the similarity modules to generate the ground truths
python module/spatial_similarity.py
python module/temporal_similarity.py
python utils/data_utils.py

# Now run the main
python main.py