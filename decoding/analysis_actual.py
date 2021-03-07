#! /usr/bin/env python3
'''Script gathers all the models whcih we have elapsed by now.'''
import os
from sklearn.pipeline import Pipeline
import decoding as dec
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA

### define parameters for decoding model
inp_dir = '/data/akitaitsev/decoding_model_bids_fmri/processed_data/'
stim_param_dir = '/data/akitaitsev/decoding_model_bids_fmri/raw_data/processed_stimuli/'
model_configs = [{'subjects': ['01']}, {'subjects': ['02']}, {'subjects': ['03']}, {'subjects': ['04']}]
out_dir = '/data/akitaitsev/decoding_model_bids_fmri/decoding_data/'



### spatial models
print('Running spatial models...')
preprocessors = [FastICA(n_components=100), FastICA(n_components=300),FastICA(n_components=300),\
    PCA(n_components=100), PCA(n_components=300),PCA(n_components=300)]

spatial_decoders = [dec.myRidge(alphas=[0,5,10], normalize=True), CCA(n_components=30), \
    dec.myRidge(alphas=[0,5,10], normalize=True),dec.myRidge(alphas=[0,5,10], normalize=True),\
    CCA(n_components=30),dec.myRidge(alphas=[0,5,10], normalize=True)] 

# create output directories for different spatial models
out_dir_spatial = ['ica100_ridge','ica300_cca30','ica300_ridge','pca100_ridge',\
    'pca300_cca30','pca300_ridge']

# run decoding models
for subject in range(len(model_configs)):
    for model in range(len(preprocessors)):
        out_dir_iter = os.path.join(out_dir, 'SM', out_dir_spatial[model]) 
        if not os.path.isdir(out_dir_iter):
            os.makedirs(out_dir_iter)
        spatial_model = Pipeline([('preprocessor',preprocessors[model]), ('decoder', spatial_decoders[model])])
        dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_configs[subject], spatial_model)



### spatial temporal models
print('Running spatial temporal models')
# create spatial-temporal decoders
lag_par = 3 # determined by GridSearch
spatial_decoders = [CCA(n_components=30),FastICA(n_components=100),PCA(n_components=100),\
    PCA(n_components=300), PCA(n_components=300), dec.myRidge(alphas=[0,5,10])]

temporal_decoders=[ dec.myRidge,dec.myRidge,dec.myRidge,CCA,dec.myRidge,dec.myRidge]

temporal_decoders_configs = [{'alphas':[0,5,10]}, {'alphas':[0,5,10]},\
    {'alphas':[0,5,10]}, {'n_components':30},\
    {alphas=[0,5,10]}, {'alphas':[0,5,10]}]

folder_names = ['cca30_ridge','ica100_ridge','pca100_ridge','pca300_cca30',\
'pca300_ridge', 'ridge_ridge']

for subject in range(len(model_configs)):
    for model in range(len(spatial_decoders)):
            print('Running STM '+folder_names[model])
            temporal_decoder = dec.temporal_decoder(temporal_decoders[model], temporal_decoders_configs[model], lag_par)
            spatial_decoder = spatial_decoders[model]
            STM = Pipeline([('spatial', spatial_decoder), ('temporal', temporal_decoder)])
            # create output dirs
            out_dir_iter = os.path.join(out_dir, 'STM', folder_names[model])
            if not os.path.isdir(out_dir_iter):
                os.makedirs(out_dir_iter)
            # run decoding model
            dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_configs[subject], decoder=STM)
