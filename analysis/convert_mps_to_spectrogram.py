'''Script converts original and predicted MPS representation into spectrogram,
computes the correlation between origianal and predicted spectrogram and 
saves these files as .pkl in the folders containing the original feature
representation.'''

import numpy as np
import joblib
import os
import mps2spectrogram_convertor as conv


orig_main_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/2'
reconstr_main_dir='/data/akitaitsev/data1/decoding_data5/'
subjects= ['sub-01','sub-02','sub-03','sub-04']
sm_models = ['pca100_ridge', 'pca300_ridge','pca300_cca30','ica300_cca30']
stm_models =['pca100_ridge','pca300_ridge','pca300_cca30','ica100_cca30'] 

def windowwise_corr(orig_spectr, reconstr_spectr, params):
    '''Computes windowwise (hop_length_mps) correlation between original and 
    reconstructed spectrogram'''
    if orig_spectr.shape != reconstr_spectr.shape:
        raise ValueError("Shapes of original and reconstructed spectrogram do not match!")
    corrs = []
    
    int_ind = params["hop_length_mps"]*(orig_spectr.shape[1]//params["hop_length_mps"]) 
    flat_orig_spectr = np.transpose(orig_spectr[:,:int_ind], (1,0)).flatten()
    flat_reconstr_spectr = np.transpose(reconstr_spectr[:,:int_ind], (1,0)).flatten()
    n_splits= flat_orig_spectr.shape[0]//(params["hop_length_mps"]*orig_spectr.shape[0])
    split_orig_spectr = np.array_split(flat_orig_spectr, n_splits)
    split_reconstr_spectr = np.array_split(flat_reconstr_spectr, n_splits) 
    for orig_wind, reconstr_wind in zip (split_orig_spectr, split_reconstr_spectr):
        corrs.append(np.corrcoef(orig_wind, reconstr_wind)[0,-1])
    corrs.append(np.corrcoef(orig_spectr[:,int_ind+1:].flatten(), reconstr_spectr[:,int_ind+1:].flatten())[0,-1])
    return np.array(corrs)

# Load the original feature representation
orig_spectr = []
phases_cum = []
for runnum in range(1,9):
    mps_path = os.path.join(orig_main_dir, ('task-aomovie_run-'+str(runnum)+'_stim.tsv.gz'))
    stim_param_path=os.path.join(orig_main_dir,('task-aomovie_run-'+str(runnum)+'_stim_parameters.json'))
    mps_phase_path=os.path.join(orig_main_dir, ('mps_phases_run-'+str(runnum)+'.tsv.gz'))
    mps, params, phases =  conv.load_data(mps_path, stim_param_path, mps_phase_path)
    spectr = conv.mps2spectr_convertor(mps, params, phases)
    orig_spectr.append(spectr)
    phases_cum.append(phases)
orig_spectr = np.concatenate(orig_spectr, axis=1)
phases_cum=np.concatenate(phases_cum, axis=0)

# convert reconstructed feature representation into spectrogram

# SM
for model in sm_models:
    for subject in subjects:
        subj_dir= os.path.join(reconstr_main_dir,'SM', model,subject)
        # if model elapsed for sbject
        if os.path.isdir(subj_dir) and len(os.listdir(subj_dir)) != 0:
            mps_path = os.path.join(subj_dir, ('reconstructed_mps_'+subject+'.pkl'))
            stim_param_path=os.path.join(orig_main_dir,('task-aomovie_run-'+str(runnum)+\
                '_stim_parameters.json'))
            mps, params,_  =  conv.load_data(mps_path, stim_param_path, None)
            reconstr_spectr = conv.mps2spectr_convertor(mps, params, phases_cum)

            ## compute correlation between flattened original and reconstructed spectrogram
            # check if predicted spectrogram is longer than original and remove "extra" part
            # of original spectrogram - original spectr may have been cropped in preprocessing
            if orig_spectr.shape[1] != reconstr_spectr.shape[1]:
                orig_spectr_ = orig_spectr[:,:reconstr_spectr.shape[1]]
            else:
                orig_spectr_ = orig_spectr
            corr = windowwise_corr(orig_spectr_, reconstr_spectr, params)
            # save data
            with open(os.path.join(reconstr_main_dir,'SM', model, subject, ('reconstructed_spectr_'\
                + subject + '.pkl')), 'wb' ) as fl:
                joblib.dump(reconstr_spectr, fl)
            with open(os.path.join(reconstr_main_dir,'SM', model, subject, ('correlations_spectr_'+\
                subject + '.pkl')), 'wb' ) as fl:
                joblib.dump(corr, fl)

# STM
for model in stm_models:
    for subject in subjects:
        subj_dir= os.path.join(reconstr_main_dir,'STM', model, subject)
        # if model elapsed for sbject 
        if os.path.isdir(subj_dir) and len(os.listdir(subj_dir)) != 0:
            mps_path = os.path.join(subj_dir, ('reconstructed_mps_'+subject+'.pkl'))
            stim_param_path=os.path.join(orig_main_dir,('task-aomovie_run-'+str(runnum)+\
                '_stim_parameters.json'))
            mps, params, _ =  conv.load_data(mps_path, stim_param_path, None)
            reconstr_spectr = conv.mps2spectr_convertor(mps, params, phases_cum)

            ## compute correlation between flattened original and reconstructed spectrogram
            # check if predicted spectrogram is longer than original and remove "extra" part
            # of original spectrogram - original spectr may have been cropped in preprocessing
            if orig_spectr.shape[1] != reconstr_spectr.shape[1]:
                orig_spectr_ = orig_spectr[:,:reconstr_spectr.shape[1]]
            else:
                orig_spectr_ = orig_spectr
            corr = windowwise_corr(orig_spectr_, reconstr_spectr, params)
            
            # save data
            with open(os.path.join(reconstr_main_dir,'STM',model, subject, ('reconstructed_spectr_' \
                + subject + '.pkl')), 'wb') as fl:
                joblib.dump(reconstr_spectr, fl)
            with open(os.path.join(reconstr_main_dir,'STM', model, subject, ('correlations_spectr_'+\
                subject + '.pkl')), 'wb') as fl:
                joblib.dump(corr, fl)
