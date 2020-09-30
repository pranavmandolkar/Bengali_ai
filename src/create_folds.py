#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 01:04:03 2020

@author: pranavmandolkar
"""

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    print(df.head())
    df.loc[:, "kfold"] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    X = df.image_id.values
    y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
    
    mskf = MultilabelStratifiedKFold(n_splits=5)
    
    for fold, (trn_idx, val_idx) in enumerate(mskf.split(X, y)):
        print("TRAIN: ", trn_idx, "VAL: ", val_idx)
        df.loc[val_idx, "kfold"] = fold
        
    print(df.kfold.value_counts())
    df.to_csv("../input/train_folds.csv", index=False)
    
