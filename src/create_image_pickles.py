#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:10:07 2020

@author: pranavmandolkar
"""

import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("../input/train_*.parquet")
    for f in files:
        df = pd.read_parquet(f)
        print(df.haed())
        image_ids = df.image_id.values
        df = df.drop("image_id", axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f"../input/image_pickles/{img_id}.pkl")
   
