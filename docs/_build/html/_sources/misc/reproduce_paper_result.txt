Reproduce paper results
=======================
You may follow the following steps to reproduce the results presented in the ACMMM2016 paper "Event Localization in Music Auto-tagging" 

Datasets
--------
Training:

* [MagnaTagATune](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)

Frame-level evaluation:

* [MedleyDB](http://medleydb.weebly.com/)

Feature extraction
------------------
scripts/extract_feats.py

Make features and annotations into experiment data
--------------------------------------------------
???

Standardize experiment data
---------------------------
scripts/standardize_data.py

Train model
-----------
scripts/train_model.py

Get thresholds
--------------
With MagnaTagATune data:

* get_thresholds.with_magnatagatune.py

With MedleyDB data:

* get_thresholds.with_medleydb.py

Evaluate model
--------------
For clip-level prediction:

* test_model.clip.py

For frame-level prediction:

* test_model.frame.py

