#! /usr/bin/env python3
'''Script to run the analysis '''
import os
from sklearn.pipeline import Pipeline
import decoding as dec
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA

### define parameters for decoding model
inp_dir = '/data/akitaitsev/decoding_model_bids/lagged/'
stim_param_dir = '/data/akitaitsev/decoding_model_bids/raw_data/processed_stimuli/'
model_configs = [{'subjects':['03']}, {'subjects':['04']}]
out_dir = '/data/akitaitsev/decoding_model_bids/decoding_data/'


### spatial temporal models
print('Running spatial temporal models')
# create spatial-temporal decoders
lag_par = 3 # determined by GridSearch
spatial_decoders = [dec.myRidge]
spatial_decoders_configs = [ {'alphas':[0,5,5]}]
spatial_decoders = dec.invoke_decoders(spatial_decoders, spatial_decoders_configs)

temporal_decoders = [dec.myRidge]
temporal_decoders_configs = [ {'alphas':[0,5,5]}]
folder_names = ['ridge_ridge']

for subject in range(len(model_configs)):
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
            dec.run_decoding(inp_dir, out_dir_iter, stim_param_dir, model_configs[subject], decoder=STM)
        folder_cntr += 1
