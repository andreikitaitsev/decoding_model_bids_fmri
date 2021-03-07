#! /usr/bin/env python3
'''Script to run the analysis using decoding.py'''
import os
from sklearn.pipeline import Pipeline
import decoding as dec
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA

### define parameters for decoding model
inp_dir = '/data/akitaitsev/decoding_model_bids/processed_data/'
stim_param_dir = '/data/akitaitsev/decoding_model_bids/raw_data/processed_stimuli/'
model_configs = [{'subjects': ['01']}, {'subjects': ['02']}, {'subjects': ['03']}, {'subjects': ['04']}]
out_dir = '/data/akitaitsev/decoding_model_bids/decoding_data/'

### spatial models
print('Running spatial models...')
preprocessor = CCA(n_components=30)
spatial_decoder = dec.myRidge(**{'alphas': [0,5,10], 'normalize':True})

# create output directories for different spatial models
out_dir_spatial = 'cca30_ridge'

# run decoding models
for subject in range(len(model_configs)):
    if not os.path.isdir(os.path.join(out_dir,'SM',out_dir_spatial)):
        os.makedirs(os.path.join(out_dir,'SM',out_dir_spatial))
    spatial_model = Pipeline([('preprocessor',preprocessor), ('decoder', spatial_decoder)])
    dec.run_decoding(inp_dir, os.path.join(out_dir,'SM',out_dir_spatial), stim_param_dir, model_configs[subject], spatial_model)
