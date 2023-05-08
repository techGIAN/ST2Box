# ST2Box
A deep representation learning framework using box architectures to capture spatiotemporal similarity relations among trajectories.

## Requirements

Please first install the requirements using:

```
pip install -r requirements.txt
```

Download some trajectory datasets such as:

* T-Drive: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
* NYC: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

You can always use your own dataset and curate it according to our formatting. See below.

## Running the Model

### Initial Setups

1. Create a directory called ```data/``` and another subdirectory ```tdrive/``` within that. In this example, we are considering the T-Drive dataset. Also include the following subdirectories in this folder: ```road/``` and ```st_traj/```.

2. Populate the ```road/``` directory with the following files: ```node.csv```, ```edge.csv```, and ```edge_weight.csv```. See steps 3-5 to see what these files and file formats are. Skip to Step 6 otherwise.

3. For ```node.csv```, this represents the nodes in the road network (road intersections). In other words, if we query ```df = pd.read_csv('node.csv')```, we expect the following result (for the first few lines at least).

```
   node         lng        lat
0     0  116.389441  39.906272
1     1  116.393070  39.906394
2     2  116.397096  39.906522
    ...
```

4. Then for ```edge.csv```, we have the following.

```
   edge  s_node  e_node       s_lng      s_lat       e_lng      e_lat       c_lng      c_lat
0     0       0    3015  116.389441  39.906272  116.389446  39.906011  116.389443  39.906142
1     1       4    7046  116.393461  39.898872  116.393316  39.901552  116.393389  39.900212
2     2    7046    3609  116.393316  39.901552  116.393080  39.906147  116.393198  39.903849
    ...
```

5. Finally for ```edge_weight.csv```:

```
   section_id  s_node  e_node      length
0           0       0    3015   28.981335
1           1       4    7046  298.308836
2           2    7046    3609  511.305049
          ...
```

6. Next is to populate the ```st_traj/``` directory with the map-matched data. Make sure to map-match the data first before doing anything else. Here are some links for map-matching:

* https://github.com/LibCity/Bigscity-LibCity
* https://github.com/categulario/map_matching
* https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/

7. The map-matched data is to be saved under ```matching_result.csv``` of ```st_traj/``` directory, with the following formatting:

```
   Traj_id                                          Node_list                                          Coor_list
0        0  [511, 510, 9220, 493, 24771, 13121, 24771, 131...  [[116.58213799999999, 40.0731091], [116.582604...
1        1  [21065, 13133, 13132, 13133, 13131, 13134, 141...  [[116.57986470000002, 40.0791802], [116.580350...
2        2  [14150, 14147, 13127, 14143, 13127, 13128, 131...  [[116.58595279999999, 40.0795623], [116.585665...
       ...
```

8. And for ```time_drop_list.csv``` under ```st_traj/``` which contains the timestamps of the trajectories:

```
   Traj_id                                          Time_list
0        0  [1201951289, 1201951289, 1201951292, 120195129...
1        1  [1201999552, 1201999552, 1201999573, 120199959...
2        2  [1202015823, 1202015823, 1202015836, 120201583...
       ...
```

9. Skip this Step 9 if you do not want to express trajectories as pathlets. But note that you can express trajectories as pathlets of length 1 by the following (but you have to do the same for the timestamps too), add such files:

```
   Traj_id       Node_list                                          Coor_list
0        0      [511, 510]  [[116.58213799999999, 40.0731091], [116.582604...
1        1     [510, 9220]  [[116.582604, 40.0754501], [116.58234099999999...
2        2     [9220, 493]  [[116.58234099999999, 40.0751418], [116.581972...
       ...
```

10. So now you have the following hierarchy in the ```data/``` directory:

```
data
|- rome
|- tdrive
|     |- road
|     |    |- edge_weight.csv
|     |    |- edge.csv
|     |    |- node.csv
|     |- st_traj
|     |    |- matching_result.csv
|     |    |- time_drop_list.csv
```

### Pathlet2Vec

Now is the time to embed the trajectories/road segments to spatiotemporal embedding vectors. Skip this entire steps and proceed to the Set2Box phase if you do not want the spatiotemporal information and just keep the raw elements. If we use pathlets, then this is the Pathlet2Vec. Otherwise, we could use road segments of the road network which is the ST2Vec that is simply identical to Fang et al. [1].

0. Modify the ```config.yaml``` file as necessary with the parameters you desire.

1. Train the model. All the commands necessary to train the model (and generate the ground truths) are bundled within the following shell script:

```
sh pathlet2vec_run.sh
```

2. Now learn the ST embeddings of the pathlets/trajectories using the following command:

```
python pathlet2vec_embed.py
```

This should create the following files in ```./embeds/``` directory.

```
train_tdrive_TP_embedding.py
valid_tdrive_TP_embedding.py
test_tdrive_TP_embedding.py
```

3. Copy and paste those three files to another directory ```./traj_embeddings/``` that you must create if you still don't have it.  Also copy and paste the ```test_st_distance.npy``` file to the directory ```./data/tdrive/st_traj/TP/```. You can run the following commands, with the last command for the final GT extractor and embedding preprocessing.

```
mkdir traj_embeddings               # ignore this line if traj_embeddings/ already exists 
mv ./embeds/train_tdrive_TP_embedding.py ./traj_embeddings/ 
mv ./embeds/valid_tdrive_TP_embedding.py ./traj_embeddings/
mv ./embeds/test_tdrive_TP_embedding.py ./traj_embeddings/
mv ./data/tdrive/ground_truth/test_st_distance.py ./data/tdrive/st_traj/TP/
sh embedding_preprocess.sh
```

### Pathlet2Box

Now, we will do Pathlet2Box that is adapted from Lee et al. [2] but for pathlets. But first ensure that the following files are found in the ```data/tdrive/st_traj/``` directory (as a result of running the last command above):

```
tdrive_roads_train.txt           # training set
tdrive_roads_valid.txt           # validation set
tdrive_roads_test.txt            # testing set
tdrive_roads_query.txt           # query set
```

Note that ```roads``` here can be replaced with ```hexes``` for hexagon blocks, or ```points_lon``` (for raw longitudinal points) and ```points_lat``` (for raw latitudinal points). Note that setting it as "points" would mean that we need both the longitudinal and latitudinal. Of course, if we want spatiotemporal information (embeddings from ST2Vec), then use ```stroads``` instead of ```roads```:

```
tdrive_stroads_train.txt           # training set
tdrive_stroads_valid.txt           # validation set
tdrive_stroads_test.txt            # testing set
tdrive_stroads_query.txt           # query set
```

All these can be fixed on the ```run.sh``` file so any modifications Now refer to ```run.sh``` for some configurations that you need to change.  Part 1 there is to train the box model. Part 2 is similarity computation. And then part 3 is calculation of evaluation scores. To run it, then we have:

```
sh run.sh
```

Note that if it cannot find some folder, then you would have to create the directory as required. If it cannot find some file, it might be stored in another directory.

## References and Citation

[1] Ziquan Fang, Yuntao Du, Xinjun Zhu, Danlei Hu, Lu Chen, Yunjun Gao, and Christian S. Jensen. 2022. "Spatio-Temporal Trajectory Similarity Learning in Road Networks". In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22). Association for Computing Machinery, New York, NY, USA, 347–356. https://doi.org/10.1145/3534678.3539375

[2] Geon Lee, Chanyoung Park and Kijung Shin, "Set2Box: Similarity Preserving Representation Learning for Sets," 2022 IEEE International Conference on Data Mining (ICDM), Orlando, FL, USA, 2022, pp. 1023-1028, doi: 10.1109/ICDM54844.2022.00125.

If you like our work or if you plan to use it, please cite our work with the following Bibtex format:

```
Bibtex format here...
```

Or you can also use this citation:

(Need to insert citation)

#### Contact

Please contact me gcalix@eecs.yorku.ca for any bugs/issues/questions you may have found on the code.
