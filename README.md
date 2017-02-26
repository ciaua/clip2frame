Clip2frame
==========

Clip2frame provides Python codes for the models in "Event Localization in Music Auto-tagging" by Jen-Yu Liu and Yi-Hsuan Yang, published in ACM Multimedia 2016.


Demo
====
http://clip2frame.ciaua.com

You can put any youtube link into the UI and see the frame-level prediction of our model.
The Gaussian filter is removed from the model to produce more sharp prediction (Gaussian is still used in training)


Scripts
=======
The following lists the scripts in the `Scripts` folder.

Training
--------
Scripts/train_model.py

Testing -- Clip level 
---------------------
Scripts/test_model.clip.py

Testing -- Frame level 
----------------------
Scripts/test_model.frame.py

Feature extraction
------------------
Scripts/extract_feats.py

Standardize data
----------------
Scripts/standardize_data.py

Network structure
-----------------
Scripts/network_structure.py

Threshods derivation
--------------------
Scripts/get_thresholds.with_magnatagatune.py
Scripts/get_thresholds.with_medleydb.py


Data for experiments in the paper
=================================

The learned parameters of the best model with Gaussian filter in the paper is included in the package:
```
data/models/model.20160309_111546.npz
```
The network structure can be found here:
Scripts/network_structure.py

This model has 188D output. The corresponding tags is here:
```
data/data.magnatagatune/tag_list.top188.txt
```

The correspondance between the 9 instruments for test in MedleyDB and the 188 tags in MagnaTagATune is here:
```
data/data.medleydb/instrument_list.medleydb_magnatagatune.top9.csv
```


The training, validation and testing features can be downloaded from the following links.

MagnaTagATune: clip-level training and testing
----------------------------------------------
http://mac.citi.sinica.edu.tw/~liu/data/exp_data.MagnaTagATune.188tags.zip

Decompress the files into a folder
```
unzip exp_data.MagnaTagATune.188tags.zip -d exp_data.MagnaTagATune
```


MedleyDB: frame-level testing
-----------------------------
Feature:
http://mac.citi.sinica.edu.tw/~liu/data/feature.MedleyDB.zip

Annotations:
http://mac.citi.sinica.edu.tw/~liu/data/annotation.medleydb.top9.zip

Data for deriving thresholds:
http://mac.citi.sinica.edu.tw/~liu/data/exp_data.MedleyDB.length_320.zip

Decompress the files
```
unzip feature.MedleyDB.zip -d feature.MedleyDB
unzip annotation.medleydb.top9.zip
```
