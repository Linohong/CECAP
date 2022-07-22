# CECAP
This source code is built upon the KAT-TSLF model in the paper
"A Novel Three-Stage Learning Framework for Low-Resource Knowledge-Grounded Dialogue Generation."
Currently, only the knowledge selection part is uploaded in this git.

## Environments
* python 3.6+
* transformers 3.2+
* NLTK
* pytorch
* language_evaluation (install from SKT project)

## Datasets 
1. Download [Wizard-of-Wikipedia](https://stuneueducn-my.sharepoint.com/personal/20151119_stu_neu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F20151119%5Fstu%5Fneu%5Fedu%5Fcn%2FDocuments%2Fshare%2FKAT%2DLSTF%2Fdataset%2Ezip&parent=%2Fpersonal%2F20151119%5Fstu%5Fneu%5Fedu%5Fcn%2FDocuments%2Fshare%2FKAT%2DLSTF&ga=1) here, then put every files in the dataset directory 
under the CECAP/dataset directory. 

2. run load_wizard.py --> this will create pre-processed files under the dataset/wizard-kat directory.

3. Download cecap_wikipedia data from [here](https://sogang365-my.sharepoint.com/:u:/g/personal/hongtaesuk_o365_sogang_ac_kr/EQiEurbmyl9Lk5v8BDhlqN8BGQtcda_umQDfijT_EkoCtg?e=6asd7j).


## Train and eval Knowledge Selection model.
```bash
./scripts/run_ks.sh
```

