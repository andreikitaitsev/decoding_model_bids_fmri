#! /usr/bin/env python3
'''Script to run the analysis using decoding_model_v6_script_based.py'''
import os
from sklearn.pipeline import Pipeline
import decoding_v6_script_based as dec
from sklearn.decomposition import PCA


## define parameters for decoding model
lag_par = 4

inp_dir = '/data/akitaitsev/data1/lagged/'
stim_param_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/'
model_config = {'subjects': ['01']}
out_dir = '/data/akitaitsev/data1/decoding_data3_script_based/'

### spatial models
print('Running spatial models...')
spatial_decoders = [dec.myRidge, dec.myRidge, dec.myCCA]

# decoders' configs
ridge_without_pca_config = {"alphas": [0,5,10]}
ridge_with_pca_config = {"alphas": [0,5,10], "var_explained": 0.9}
cca_config = {"n_components":300}
spatial_decoders_configs = [ridge_without_pca_config, ridge_with_pca_config,\
    cca_config]

# invoke decoders
spatial_models = dec.invoke_decoders(spatial_decoders, spatial_decoders_configs)

# create output directories for different spatial models
out_dir_spatial = ['ridge','ridge_pca0.9','cca300']

# run decoding models
for model_num  in range(len(spatial_decoders)):
    out_dir_iter = os.path.join(out_dir, 'SM', out_dir_spatial[model_num]) 
    if not os.path.isdir(out_dir_iter):
        print('Creating output directory ' + out_dir_iter + '...')
        os.makedirs(out_dir_iter)
    dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_config, \
        spatial_models[model_num])

### spatial temporal models
print('Running spatial temporal models')


# spatial_temporal models
# create spatial-temporal decoder
spatial_decoders = [PCA, dec.myRidge, dec.myRidge, dec.myCCA]
spatial_decoders_configs = [ {'n_components':200},{'alphas':[0,5,5]}, {'alphas':[0,5,5],\
    'var_explained':0.9}, {'n_components':200}]
spatial_decoders = dec.invoke_decoders(spatial_decoders, spatial_decoders_configs)

temporal_decoders = [dec.myRidge, dec.myRidge, dec.myCCA]
temporal_decoders_configs = [ {'alphas':[0,5,5]}, {'alphas':[0,5,5], 'var_explained': 0.9},\
    {'n_components':200}]
folder_names = ['PCA_ridge','PCA_ridgePCA0.9', 'PCA_cca200', \
    'ridge_ridge','ridge_ridgePCA0.9','ridge_cca200',\
    'ridgePCA0.9_ridge', 'ridgePCA0.9_ridgePCA0.9', 'ridgePCA0.9_cca200',\
    'cca200_ridge', 'cca200_ridgePCA0.9', 'cca200_cca200']

folder_cntr = 0 # folder counter
for sp_dec in range(len(spatial_decoders)):
    for tmp_dec in range(len(temporal_decoders)):
        temporal_decoder = dec.temporal_decoder(temporal_decoders[tmp_dec], temporal_decoders_config[tmp_dec], lag_par)
        spatial_decoder = spatial_decoders[sp_dec]
        STM = Pipeline([('spatial', spatial_decoder), ('temporal', temporal_decoder)])
        # create output dirs
        out_dir_iter = os.path.join(out_dir, 'STM', folder_names[folder_cntr])
        if not os.path.isdir(out_dir_iter):
            print('Creating output directory ' + out_dir_iter + '...')
            os.makedirs(out_dir_iter)
        # run decoding model
        dec.run_decoding(inp_dir, out_dir, stim_param_dir, model_config, decoder=STM)

