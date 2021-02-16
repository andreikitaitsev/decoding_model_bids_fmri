#! /usr/bin/env python3
'''Script to run the analysis using decoding_model_v7.py'''
import os
from sklearn.pipeline import Pipeline
import decoding_v7 as dec
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA

### define parameters for decoding model
lag_par = 4

inp_dir = '/data/akitaitsev/data1/lagged/'
stim_param_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/'
model_config = {'subjects': ['01']}
out_dir = '/data/akitaitsev/data1/decoding_data5/'



### spatial models
print('Running spatial models...')
preprocessors = [PCA(n_components=300), FastICA(n_components=300)]
spatial_decoders = [dec.myRidge, CCA, CCA]
ridge_config = {"alphas": [0,5,10]}
cca30_config = {'n_components':30}
cca100_config = {'n_components':100}
spatial_decoders_configs = [ridge_config, cca30_config, cca100_config]
spatial_models = dec.invoke_decoders(spatial_decoders, spatial_decoders_configs)

# create output directories for different spatial models
out_dir_spatial = ['pca300_ridge','ica300_ridge', 'pca300_cca30','ica300_cca30', 'pca300_cca100', 'ica300_cca100']

# run decoding models
cntr = 0
for model_num  in range(len(spatial_decoders)):
    for preprocessor in range(len(preprocessors)):
        out_dir_iter = os.path.join(out_dir, 'SM', out_dir_spatial[cntr]) 
        if not os.path.isdir(out_dir_iter):
            os.makedirs(out_dir_iter)
        spatial_model = Pipeline([('preprocessor',preprocessors[preprocessor]), ('decoder', spatial_models[model_num])])
        dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_config, spatial_models[model_num])
        cntr += 1


### spatial temporal models
print('Running spatial temporal models')
# create spatial-temporal decoders
lag_par = 3 # determined by GridSearch
spatial_decoders = [PCA, FastICA, dec.myRidge, CCA]
spatial_decoders_configs = [ {'n_components':300}, {'n_components':100}, {'alphas':[0,5,5],\
    'var_explained':0.9}, {'n_components':100}]
spatial_decoders = dec.invoke_decoders(spatial_decoders, spatial_decoders_configs)

temporal_decoders = [dec.myRidge, CCA]
temporal_decoders_configs = [ {'alphas':[0,5,10]},  {'n_components':30}]
folder_names = ['pca300_ridge', 'pca300_cca30', \
    'ica100_ridge', 'ica100_cca30',
    'ridge0.9_ridge','ridge0.9_cca30',\
    'cca100_ridge', 'cca100_cca30']

folder_cntr = 0 # folder counter
for sp_dec in range(len(spatial_decoders)):
    for tmp_dec in range(len(temporal_decoders)):
        print('Running STM '+folder_names[folder_cntr])
        temporal_decoder = dec.temporal_decoder(temporal_decoders[tmp_dec], temporal_decoders_configs[tmp_dec], lag_par)
        spatial_decoder = spatial_decoders[sp_dec]
        STM = Pipeline([('spatial', spatial_decoder), ('temporal', temporal_decoder)])
        # create output dirs
        out_dir_iter = os.path.join(out_dir, 'STM', folder_names[folder_cntr])
        if not os.path.isdir(out_dir_iter):
            os.makedirs(out_dir_iter)
        # run decoding model
        dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_config, decoder=STM)
        folder_cntr += 1
