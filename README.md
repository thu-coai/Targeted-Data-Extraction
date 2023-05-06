# Targeted-Data-Extraction

Code for the ACL 2023 paper "Ethicist: Targeted Training Data Extraction Through Loss Smoothed Soft Prompting and Calibrated Confidence Estimation".


## Environment
```
conda env create -f py38.yaml
```

## Run

The complete data in the `datasets` folder contains 15,000 samples. The first 14,000 samples are randomly split into the training set (12,600 samples) and the validation set (1,400 samples). The last 1000 samples make up the test set.

### 1.Prompt tuning
```
cd prompt
bash train.sh
```
Please change the following path to your own path:
- basemodel_path in train.sh

### 2.Sample suffixes
```
cd prompt
bash gen.sh
```
Please change the following path to your own path:
- basemodel_path in gen.py
- ckpt path in gen.py

### 3.Obtain final predictions and compute metrics
```
python score_multiple_gen.py
```
Please change the following path to your own path:
- tokenizer path in score_multiple_gen.py
- resdir in score_multiple_gen.py (the result path in step2)
