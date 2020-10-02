# APSI
This is the source code for the EMNLP 2020 paper: Analogous Process Structure Induction for Sub-event Sequence Prediction


### Requirement
```
Python 3.6
allennlp 1.0.0
spacy 2.2.2
torch 1.6.0
```

### Usage Instruction


##### Preparation
Download the [cleaned dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/ESTDjgtYTGdFsFIpRAPaVX0BIhrfdaxMNBHN9Q3KKRW7BA?e=0gbeFe) and move the intrinsic_dataset and extrinsic_dataset folders to the current location.

##### Intrinsic Evaluation
(1) Induce the abstractive representation with ```python abstractive_representation_induction.py```

(2) Evaluate APSI with ```python Evaluate_APSI.py```

#####  Extrinsic Evaluation
(1) Train the event LM with ```python event_LM_training.py```

(2) Test the performance of the LM with ```python event_prediction_evaluation.py```

### Bib file

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. Please kindly cite our paper:
```
@inproceedings{APSIHMHYD,
  author    = {Hongming Zhang and
               Muhao Chen and
               Haoyu Wang and
               Yangqiu Song and
               Dan Roth},
  title     = {Analogous Process Structure Induction for Sub-event Sequence Prediction.},
  booktitle = {Proceedings of EMNLP 2020},
  year      = {2020}
}
```

### Others
If you have some questions about the code, you are welcome to open an issue or send me an email, I will respond to that as soon as possible.
