#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 08:59:33 2021

@author: forustous
"""
#%%
SEED = 0

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.utils.class_weight import compute_sample_weight

df=pd.read_csv("/home/danssa/proj_ua/data/60model_train_share.csv")

X_train = df.drop('eGFR_ab', axis=1)
y_train = df['eGFR_ab'].astype("int64")

weights = compute_sample_weight(class_weight="balanced", y=y_train)

#%% 
# optuna

automl = AutoML(mode="Optuna", ml_task="binary_classification", 
    algorithms=["CatBoost"], eval_metric='auc',
    optuna_time_budget=10*60,
    total_time_limit = 24*3600,
    golden_features = False, 
    features_selection = False,
    train_ensemble= True,
    stack_models = 'auto',
    random_state=SEED, results_path="optuna")

automl.fit(X_train, y_train, weights)

# %%
# explain

automl = AutoML(mode="Explain", ml_task="binary_classification", 
    algorithms=["Baseline", "CatBoost", "Xgboost", "Random Forest", "Extra Trees", "LightGBM", "Neural Network"], 
    eval_metric='auc',
    train_ensemble= False, 
    random_state=SEED,
    results_path="explain-wt")

automl.fit(X_train, y_train, weights)

# %%
# Perform

automl = AutoML(mode="Perform", ml_task="binary_classification", 
    algorithms=["CatBoost", "Xgboost"], 
    eval_metric='auc',
    golden_features=False,
    features_selection=False,
    train_ensemble = False,
    stack_models = False, 
    random_state=SEED,
    results_path="perform")

automl.fit(X_train, y_train, weights)

# %%

