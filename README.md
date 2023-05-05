# ST2Box
A deep representation learning framework using box architectures to capture spatiotemporal similarity relations among trajectories.

## Requirements

I will list the requirements here soon.

## Running the Model

Please first install the requirements using:

```
pip install -r requirements.txt
```

Download some trajectory datasets such as:

* T-Drive: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
* Rome: https://crawdad.org/roma/taxi/20140717
* Porto: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
* NYC: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

You can always use your own dataset and curate it according to our formatting.

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

### ST2Vec Phase

Now is the time to embed the trajectories/road segments to spatiotemporal embedding vectors. Skip this entire steps and proceed to the Set2Box phase if you do not want the spatiotemporal information and just keep the raw elements. Note that this becomes Pathlet2Vec if we consider pathlets instead of road segments. But we are mostly considering the road segments (length-1 pathlets) in this phase.

1. First preprocess the data:

```
python preprocess.py
```

2. Next is to generate the ground truths for both spatial and temporal. Note that this can take a long time, depending on the trajectory dataset and road network

```
python spatial_similarity.py
python temporal_similarity.py
```

3. Next is to run the ```data_utils.py``` which allows spatiotemporal ground truth generation.

```
python data_utils.py
```

4. Now train the model with:

```
python main.py
```

Ensure that the following is set in ```main.py``` before training:

```
load_model_name = None              # If you have a model or optimizer saved from checkpoint, replace None here.
load_optimizer_name = None 
STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)
```

5. Re-run Step 4 command once you have already have your model ready to obtain the embeddings. But first, replace the ```with torch.no_grad()``` block of ```def ST_eval(self, load_model=None)``` method in ```Trainer.py``` with:

```
with torch.no_grad():
    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='train')
    embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                   test_traj=list(vali_node_list),
                                                   test_time=list(vali_d2vec_list),
                                                   test_batch=self.test_batch)
    test_method.test_model(embedding_vali, typ='train', isvali=True)

    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
    embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                   test_traj=list(vali_node_list),
                                                   test_time=list(vali_d2vec_list),
                                                   test_batch=self.test_batch)
    test_method.test_model(embedding_vali, typ='valid', isvali=True)

    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
    embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                   test_traj=list(vali_node_list),
                                                   test_time=list(vali_d2vec_list),
                                                   test_batch=self.test_batch)
    test_method.test_model(embedding_vali, typ='test', isvali=True)
    exit()
```

Also replace the ```def test_model(embedding_set, isvali=False):``` with ```def test_model(embedding_set, typ, isvali=False):``` in ```test_method.py``` and adding the following lines after the method signature:

```
embedding_set = embedding_set.data.cpu().numpy()
np.save('./embeds/{}_nyc_TP_embedding.npy'.format(typ), embedding_set)
return
```
Next is to comment out the ```STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)``` and then uncomment ```STsim.ST_eval(load_model=load_model_name)```, where ```load_model``` should be the name of your model (and not ```None```). Finally, ensure the ```./embeds/``` directory is present before running:

```
python main.py
```

6. After Step 5, ensure you have the following files in ```./embeds/``` directory.

```
train_tdrive_TP_embedding.py
valid_tdrive_TP_embedding.py
test_tdrive_TP_embedding.py
```

7. Copy and paste those three files to another directory ```./traj_embeddings/``` that you must create if you still don't have it. 

8. Copy and paste the ```test_st_distance.npy``` file to the directory ```./data/tdrive/st_traj/TP/```

9. Run the following commands:

```
python gt_extractor.py --dataset tdrive --sim TP
python traj_emb_preprocess.py -- dataset tdrive --sim TP --rounder 4
```



### Set2Box Phase

Now, we will do Set2Box. But first ensure that the following files are found in the ```data/tdrive/st_traj/``` directory (as a result of running the ```traj_emb_preprocess.py```):

```
tdrive_roads_train.txt           # training set
tdrive_roads_valid.txt           # validation set
tdrive_roads_test.txt            # testing set
tdrive_roads_query.txt           # query set
```

where the first line of the file will be ```N   M``` (separated by a tab), where there are a total of ```N``` distinct elements in the dataset T-Drive and ```M``` sets (trajectories) in the [training/validation/testing/query] text files. Then all the rest of the lines are tab-delimeted, where each line is a set (the trajectory) and each item in that line is an element. Note that all elements have to be strictly less than ```N``` and that there are no gaps. For example, if ```N = 100``` then the elements [0,...,99] must exist at least once. If not, then you can fix it for instance by using a lower ```N``` value and fill in the missing gaps by some sort of mapping or transformation. And finally ```M``` has to be correct in the number of sets. Otherwise an error is thrown.

Note that ```roads``` here can be replaced with ```hexes``` for hexagon blocks, or ```points_lon``` (for raw longitudinal points) and ```points_lat``` (for raw latitudinal points). Note that setting points would mean that we need both the longitudinal and latitudinal. Of course, if we want spatiotemporal information (embeddings from ST2Vec), then use ```stroads``` instead of ```roads```:

```
tdrive_stroads_train.txt           # training set
tdrive_stroads_valid.txt           # validation set
tdrive_stroads_test.txt            # testing set
tdrive_stroads_query.txt           # query set
```

Now refer to ```run.sh``` for some configurations that you need to change.  Part 1 there is to train the box model. Part 2 is similarity computation. And then part 3 is calculation of evaluation scores. To run it, then we have:

```
sh run.sh
```

## References and Citation

(Need to provide the references appropriately)

If you like our work or if you plan to use it, please cite our work with the following Bibtex format:

```
Bibtex format here...
```

Or you can also use this citation:

(Need to insert citation)

#### Contact

Please contact me gcalix@eecs.yorku.ca for any bugs/issues/questions you may have on the code.
