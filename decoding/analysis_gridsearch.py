
#! /usr/bin/env python3
'''Script to run the analysis using decoding_model_v6_script_based.py'''
import os
from sklearn.pipeline import Pipeline
import decoding_v7 as dec
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GridSearchCV
### define parameters for decoding model
lag_par = 4
inp_dir = '/data/akitaitsev/data1/lagged/'
stim_param_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/'
model_config = {'subjects': ['01'], 'runs':[[1]]}
out_dir = '/data/akitaitsev/data1/decoding_data3_script_based/'



### spatial models
print('Running spatial models...')
# parameters for gridsearch
params = {'alphas':[[0,1,2],[2,3,2] ],'var_explained':[0.6,0.7]}
spatial_decoder = dec.myRidge()
spatial_models = [GridSearchCV(spatial_decoder, params, scoring='r2')]
# create output directories for different spatial models
out_dir_spatial = ['ridge_pca_gridsearch']

# run decoding models
for model_num  in range(len(spatial_models)):
    out_dir_iter = os.path.join(out_dir, 'SM_draft_timer_test', out_dir_spatial[model_num]) 
    if not os.path.isdir(out_dir_iter):
        os.makedirs(out_dir_iter)
    # wrap models in Pipeline to use in cross_val_predict
    spatial_model = Pipeline([(out_dir_spatial[model_num], spatial_models[model_num])])
    dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_config, spatial_model)
