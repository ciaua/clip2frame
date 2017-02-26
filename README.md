Clip2frame
==========

Clip2frame provides Python codes for the models in "Event Localization in Music Auto-tagging" by Jen-Yu Liu and Yi-Hsuan Yang, published in ACM Multimedia 2016


Usage
=====
The `Scripts` folder contains scripts for model training and evaluation.





Data for experiments in the paper
=================================
The training, validation and testing features can be downloaded from the following links.

MagnaTagATune: clip-level training and testing
----------------------------------------------
http://mac.citi.sinica.edu.tw/~liu/data/exp_data.MagnaTagATune.188tags.zip

Decompress the files in to a folder
```
unzip exp_data.MagnaTagATune.188tags.zip -d exp_data.MagnaTagATune
```


MedleyDB: frame-level testing
-----------------------------
Feature:
http://mac.citi.sinica.edu.tw/~liu/data/feature.MedleyDB.zip

Annotations:
http://mac.citi.sinica.edu.tw/~liu/data/annotation.medleydb.top9.zip

Decompress the files
```
unzip feature.MedleyDB.zip -d feature.MedleyDB
unzip annotation.medleydb.top9.zip
```




Demo
-----
http://clip2frame.ciaua.com

You can put any youtube link into the UI and see the frame-level prediction of our model.
The Gaussian is removed from the model to produce more sharp prediction (Gaussian is still used in training)

