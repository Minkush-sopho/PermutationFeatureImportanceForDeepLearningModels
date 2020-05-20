# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:11:53 2020

@author: Minkush Manuja
"""
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
def scorer(estimator, X, y,scoring=None):
    """
        Need to implement this function
        Return the predicted scoring metric as per "scoring"
        scoring may take values such as "accuracy" ,"AUC", etc according to which the corresponding
        value is returned
        
    """
    prediction = estimator.predict(X)
    prediction = prediction.argmax(axis=1)
    y = y.argmax(axis=1) #  iff "y" is one-hot 

    if scoring=="accuracy":
        return accuracy_score(y, prediction)
                
    elif scoring=="AUC":
        return roc_auc_score(y, prediction)
    
    elif scoring == "log_loss":
        return log_loss(y, prediction)

######################################## DYNAMICS IMPORTANCE ###########################3

def generate_new_X_DI(X, features_list, permute_ts):
    X_permuted = copy.deepcopy(X)
    
    for window in range(X.shape[0]):
        for ts in range(X.shape[1]):
            for feature_idx in features_list:
                X_permuted[window][ts][feature_idx] = X[window][permute_ts[ts]][feature_idx]
                
    return X_permuted

def calculate_dynamics_importance(estimator, X, y, n_repeats, features_list, scoring = None):
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    permute_ts = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        permute_ts[i] = i
    
    for n_round in range(n_repeats):
        permute_ts = np.random.permutation(permute_ts)
        X_permuted = generate_new_X_DI(X, features_list, permute_ts)
        score = scorer(estimator, X_permuted, y, scoring)
        scores[n_round] = score
        
    return scores


def dynamics_importance(estimator, X, y, n_repeats, features_list, scoring = None):
    
    baseline_score = scorer(estimator,X,y,scoring)
    scores = calculate_dynamics_importance(estimator, X, y, n_repeats, features_list, scoring)
    deviations = baseline_score - scores
    
    return deviations


########################### PERMUTATION IMPORTANCE ################################3
        
def generate_new_X(X, permute_idx, col_idx):
    X_permuted = copy.deepcopy(X)
    for i in range(X.shape[0]):
        for ts in range(X.shape[1]):
            X_permuted[i][ts][col_idx] = X[int(permute_idx[i])][ts][col_idx]
    
    return X_permuted


def calculate_perm_imp(estimator, X, y, col_idx, n_repeats,scoring=None):
    X_permuted = copy.deepcopy(X)
    scores = np.zeros(n_repeats)
    permute_idx = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        permute_idx[i] = i
    
    for n_round in tqdm(range(n_repeats)):
        permute_idx = np.random.permutation(permute_idx)
        X_permuted = generate_new_X(X,permute_idx,col_idx)
        feature_score = scorer(estimator, X_permuted, y,scoring)        
        scores[n_round] = feature_score
        
    return scores
        
    
    

def make_dict(mean, std, raw):
    ret_stats = {}
    ret_stats["importances_mean"] = mean
    ret_stats["importances_std"] = std
    ret_stats["importances_raw"] = raw
    
    return ret_stats


def permutation_imp_resnet(estimator, X, y, n_repeats = 5, scoring = None):
    """ A custom Permutation Feature Importance Function for ResNet
    
    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted` and is compatible
        with :term:`scorer`.
    X : ndarray or DataFrame
        Data on which permutation importance will be computed.
    y : Targets for supervised or `None` for unsupervised.
    
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.
    n_repeats : int, default=5
        Number of times to permute a feature.

    """
    baseline_score = scorer(estimator, X,y,scoring)

    scores = []
    
    for col_idx in range(X.shape[2]):
        score = calculate_perm_imp(estimator, X,y,col_idx, n_repeats, scoring)
        scores.append(score)
    importances = baseline_score - np.array(scores)
    ret_stats = {}
    ret_stats["importances_mean"] = np.mean(importances,axis=1)
    ret_stats["importances_std"] = np.std(importances,axis = 1)
    ret_stats["importances_raw"] = importances
    
    return ret_stats

    

    