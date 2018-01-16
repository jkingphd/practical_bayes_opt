from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_boston

import pandas as pd
import numpy as np
import dill
import time

seed = 101 # Lucky seed
    
def score_model(**params):
    model = RandomForestRegressor(n_estimators=int(params['n_estimators']),
                                  min_samples_split=int(params['min_samples_split']),
                                  max_features=params['max_features'],
                                  n_jobs=1)
    return cross_val_score(model, X_train, y_train, scoring=neg_rmsle, cv=3).mean()
    
neg_rmsle = make_scorer(rmsle, greater_is_better=False)

if __name__ == '__main__':
    X_train = load_dill('mercari.dill')['X_train']
    y_train = load_dill('mercari.dill')['y_train']

    # # Try grid search
    
    # gs_params = {'n_estimators':[10, 20, 50, 100],
                 # 'min_samples_split':[2, 5, 10, 20],
                 # 'max_features':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

    # gs = GridSearchCV(estimator=RandomForestRegressor(random_state=seed),
                      # param_grid=gs_params,
                      # scoring=neg_rmsle,
                      # n_jobs=6,
                      # refit=False,
                      # cv=3,
                      # verbose=3,
                      # return_train_score=True)

    # start = time.time()
    # gs.fit(X_train, y_train)
    # end = time.time()
    # print('Time to perform grid search: %0.2fs' % (end - start))
    # gs_time = end - start
    
    # Try Bayesian Optimization (initialize with points from random search)
    rs_params = {'n_estimators':[10, 20, 50, 100],
                 'min_samples_split':[2, 5, 10, 20],
                 'max_features':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    rs = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=seed),
                            param_distributions=rs_params,
                            n_iter=5,
                            scoring=neg_rmsle,
                            n_jobs=6,
                            cv=3,
                            verbose=3,
                            refit=False,
                            return_train_score=True)

    start = time.time()
    rs.fit(X_train, y_train)
    dump_dill('rs_mercari_full.dill', rs)
    
    params_init = {}
    params_init.update({'target': rs.cv_results_['mean_test_score']})
    for key in rs_params.keys():
        params_init[key] = [val[key] for val in rs.cv_results_['params']] 

    params = {'n_estimators':(10,100), 'min_samples_split':(2,20), 'max_features':(0.5,1.0)}
    bo = BayesianOptimization(score_model, pbounds=params, verbose=1)
    bo.initialize(params_init)
    bo.maximize(init_points=0, n_iter=10, acq='ucb')
    end = time.time()
    print('Time to perform Bayesian optimization: %0.2fs' % (end - start))
    bo_time = end - start
    
    # dump_dill('mercari_result_full.dill', {'gs':gs, 'gs_time':gs_time, 'bo':bo, 'bo_time':bo_time})
    dump_dill('mercari_result_full.dill', {'bo':bo, 'bo_time':bo_time})