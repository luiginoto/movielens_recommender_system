#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .ALS import CustomALS
import time

def ALSValidation(X_train, X_val, rank_vals = [10], regParam_vals = [0.1], maxIter_vals = [10], metric_val= 'meanAveragePrecision',\
                                 k_val = 100, verbose = True):
    best_score = 0
    best_als_model = None
    for rank in rank_vals:
        for regParam in regParam_vals:
            for maxIter in maxIter_vals:
                if verbose:
                    print("Fitting ALS model given parameters: Rank: {rank} | regParam: {regParam} | maxIter: {maxIter} ".format(
                    rank=rank, regParam=regParam, maxIter=maxIter))
                als = CustomALS(rank = rank, regParam=regParam, maxIter=maxIter)
                start = time.time()
                als.fit(X_train)
                end = time.time()
                tot_time = end - start
                als_metrics_val = als.evaluate(X_val)
                if metric_val == 'meanAveragePrecision':
                    val_score = als_metrics_val.meanAveragePrecision
                    metric_val = 'MAP'
                elif metric_val == 'meanAveragePrecisionAtK':
                    val_score = als_metrics_val.meanAveragePrecisionAt(k_val)
                    metric_val = 'MAP'
                elif metric_val == 'ndcgAtK':
                    val_score = als_metrics_val.ndcgAt(k_val)
                    metric_val = 'NDCG'
                elif metric_val == 'precisionAtK':
                    val_score = als_metrics_val.precisionAt(k_val)
                    metric_val = 'PRECISION'
                elif metric_val == 'recallAtK':
                    val_score = als_metrics_val.recallAt(k_val)
                    metric_val = 'RECALL'
            
                if verbose:
                    print('Score for ALS model given these parameters using {m}@{k}: {s}'.format(m = metric_val, k = k_val, s = val_score))
                    print('------------------------------------------------------------------------------------------------------------------------------')
            
                if val_score > best_score:
                    best_score = val_score
                    best_als_model = als
   
    print('==============================================================================================================================')
    print("Best ALS model given parameters using {m}@{k}: Rank: {rank} | regParam: {regParam} | maxIter: {maxIter}".format(
        rank=best_als_model.rank, regParam=best_als_model.regParam, maxIter=best_als_model.maxIter, m = metric_val, k = k_val))
    print("Best ALS model score given these parameters: ", best_score)
    print("Average model fitting time for the given dataset: ",tot_time)
    return best_als_model
    
