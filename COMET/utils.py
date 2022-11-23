import os
import torch
import random

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def competition_metric(true, pred):
    return f1_score(true, pred, average='macro')


def drop_columns_by_none_ratio(df, threshold=0.3):
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    remove_list = list()

    for c in df.columns:
        if c == 'Target':   continue
        train_none_ratio = len(df.loc[df[c] == 'none']) / len(df)
        
        if train_none_ratio > threshold:
            remove_list.append(c)

    df.drop(remove_list, axis=1, inplace=True)
    
    return df, remove_list