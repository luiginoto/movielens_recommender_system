#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from popularity import PopularityBaseline

def PopularityBaselineValidation(X_train, X_val, damping_vals, threshold_vals = [None], metric_val= 'meanAveragePrecision',\
                                 k_val = 100, verbose = True):
    best_score = 0
    best_baseline_model = None
    for d in damping_vals:
        for t in threshold_vals:
            if verbose:
                print("Fitting PopularityBaseline model given parameters: Damping {d} | Threshold: {t}".format(
                    d=d, t=t))
            baseline = PopularityBaseline(threshold = t, damping = d)
            baseline.fit(X_train)
            baseline_metrics_val = baseline.evaluate(baseline.results, X_val)
            if metric_val == 'meanAveragePrecision':
                val_score = baseline_metrics_val.meanAveragePrecision
                metric_val = 'MAP'
            elif metric_val == 'meanAveragePrecisionAtK':
                val_score = baseline_metrics_val.meanAveragePrecisionAt(k_val)
                metric_val = 'MAP'
            elif metric_val == 'ndcgAtK':
                val_score = baseline_metrics_val.ndcgAt(k_val)
                metric_val = 'NDCG'
            elif metric_val == 'precisionAtK':
                val_score = baseline_metrics_val.precisionAt(k_val)
                metric_val = 'PRECISION'
            elif metric_val == 'recallAtK':
                val_score = baseline_metrics_val.recallAt(k_val)
                metric_val = 'RECALL'
            
            if verbose:
                print('Score for PopularityBaseline model given these parameters using {m}@{k}: {s}'.format(m = metric_val, k = k_val, s = val_score))
                print('------------------------------------------------------------------------------------------------------------------------------')
            
            if val_score > best_score:
                best_score = val_score
                best_baseline_model = baseline
   
    print('==============================================================================================================================')
    print("Best PopularityBaseline model given parameters using {m}@{k}: Damping {d} | Threshold: {t}".format(
        d=d, t=t, m = metric_val, k = k_val))
    print("Best PopularityBaseline model score given these parameters: ", best_score)
    return best_baseline_model
    
