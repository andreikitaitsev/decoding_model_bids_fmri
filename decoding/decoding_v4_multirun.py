#! /usr/bin/env python3
import numpy as np
import copy
from encoding import  ridge_gridsearch_per_target
from sklearn.cross_decomposition import CCA
import sys
import joblib
import os
from sklearn.cross_decomposition import CCA
sys.path.append('/data/akitaitsev/data1/code/decoding/')
from encoding import product_moment_corr, ridge_gridsearch_per_target
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
import sklearn.metrics 
import json
import matplotlib.pyplot as plt

### Helper fucntions 

def reduce_dimensionality(X, var_explained, examine_variance=False, **kwargs):
    ''' Function applys pca with user specified variance explained to the input data
    Inputs:
        X - 2d numpy array (samples/voxels)
        var_explained - float, variabce explained
        examine_variance - logical, whether to run function in interactive mode to manually determine number 
        of components to leave by variance explained plot. Default =Fasle
        kwargs - kwargs for sklearn pca obejct 
    Outputs:
        X_backproj - reduced rand data
        pca - pca object trained (fitted) on the supplied data
        comp2leave - number of components left
    '''   
    
    from sklearn.decomposition import PCA
    pca=PCA(**kwargs)
    pca.fit(X)
    vars_exp = pca.explained_variance_ratio_
    vars_exp = np.cumsum(vars_exp)
    #[vars_exp_[i]+vars_exp_[i-1] for i in range(vars_exp_.shape[0]) if i !=0]
    comp2leave= None 
    if examine_variance:
        fig, ax = plt.subplots()
        ax = fig.add_subplot(111)
        ax.plot(vars_exp*100, '--ob')
        fig.suptitle('Scree plot of variance explained')
        ax.set_xlabel('Component number')
        ax.set_ylabel('Variance explained, %')
        plt.show(block = False)

        comp2leave=int(input('Enter number of components to leave: '))
        pca = PCA(n_components=comp2leave)
        pca.fit(X)
    else:
        comp2leave = vars_exp[vars_exp<var_explained].shape[0]
    pca=PCA(n_components=comp2leave)
    pca.fit(X)
    X_backproj = pca.transform(X)
    return X_backproj, pca, comp2leave

def compute_correlation(real_mps, reconstructed_mps, **kwargs):
    ''' Fucntion computes correlation between real (extracted MPS and MPS reconstruced indecoding model for every time sample.
    Inputs:
        real_mps -          numpy 2d array(times, features) 
        reconstructed_mps - numpy 2d array(times, features) 
    Outputs:
        correlations - 1d numpy array of correlation values (1 per mps timepoins from test_inds).'''
    correlations =[]
    for time in range(reconstructed_mps.shape[0]):
        correlations.append(np.corrcoef(real_mps[time,:], reconstructed_mps[time,:])[0,-1])
    return np.reshape(np.array(correlations), (-1,1))

def plot_score_across_features(score, score_name, mps_time, mps_freqs):
    ''' Function plots any sklearn score across features (1 value per feature over different MPSs) 
        as imshow and returns figure handle
    Inputs:
        score - 2d numpy array of reshaped score
        score_name - str, name of the score used
        mps_time - list of strings of mps_time from stimulus parameters json file (output of wav_files....py)
        mps_freqs - list of strings of mps_freqs from stimulus parameters json file (output of wav_files....py)
    Outputs:
        fig - figure handle '''
    fig, ax = plt.subplots()
    fig.suptitle(score_name + ' across different features')
    im = ax.imshow(score, origin='lower', aspect='auto')
    # transform mps_time and mps_freqs from list of strings to np 1d array to allow indexing
    mps_time = np.array([float(el) for el in mps_time])
    mps_freqs = np.array([float(el) for el in mps_freqs])
    # x labels
    x_inds = np.around(np.linspace(0, len(mps_time), 10, endpoint=False)).astype(int)
    ax.set_xticks(x_inds)
    ax.set_xticklabels(mps_time[x_inds])
    ax.set_xlabel('modulation/s')
    # y labels
    y_inds = np.around(np.linspace(0, len(mps_freqs), 10, endpoint=False)).astype(int)
    ax.set_yticks(y_inds)
    ax.set_yticklabels(mps_freqs[y_inds])
    ax.set_ylabel('modulation/Hz')
    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(score_name)
    return fig

def plot_score_across_mps(score, score_name, ylim = None):
    '''Fucntion plots score across different MPSs (1 value per MPS) as curve with plt.plot.
    Inputs:
        score - 1d numpy array of score (1 value per MPS)
        score_name - str, name of the score
        ylim - list of 2 floats; Default = None
     Outputs:
        fig - figure handle
     '''
    fig, ax = plt.subplots()
    fig.suptitle(score_name + 'across different MPSs')
    ax.plot(score)
    ax.set_xlabel('MPS index')
    ax.set_ylabel(score_name + 'per MPS')
    if ylim != None:
       ax.set_ylim([ylim[0],ylim[1]])
    return fig

def reshape_mps(mps, mps_shape):
    ''' Fucntion reshapes (flattened) MPS array into its original form (mod/s, mod/Hz, n_mps) in accordance with
    mps_shape from stimulus json parameters file
    Inputs:
        mps - 2d numpy array of shape (n_mps, n_features) - output of make_X_Y function for a single wav file
        mps_shape - tuple(mod/s, mod/Hz) - shape of a single MPS stored in stimulus parameters json file
    Outputs:
        reshaped_mps - 3d numpy array of shape (mod/s, mod/Hz, n_mps) - same shape as original MPS
    '''
    #resahpe to shape (n_mps, mod/s, mod/Hz)
    reshaped_mps = np.reshape(mps, (-1, mps_shape[0], mps_shape[1]))
    # transpose to shape (mod/s, mod/Hz, n_mps)
    reshaped_mps = np.transpose(reshaped_mps, (1,2,0))
    return reshaped_mps

def plot_mps_and_reconstructed_mps(original_mps, reconstructed_mps, mps_time, mps_freqs, fig_title=None, **kwagrs):
    ''' Fucntion plots real and reconstruced MPS on different subplots of one figure
    using matplotlib imshow function
    Inputs:
        original_mps      - numpy 2d array, reshaped to original form original mps
        reconstructed_mps - numpy 2d array, reshaped to original form reconstructed mps
        mps_time          - list or numpy array of mps_time labels (output of feature extractor)
        mps_freqs         - list or numpy array of mps_freqs labels (output of feature extractor)
        fig_title         - str, optional, title of the figure (Default=None)
        kwargs            - arguments for imshow function
    Outputs:
        fig - figure handle'''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9)) 
    im1 = ax1.imshow(original_mps, origin='lower', aspect='auto')
    im2 = ax2.imshow(reconstructed_mps, origin='lower', aspect='auto') 
    # transform mps_time and mps_freqs from list of strings to np 1d array to allow indexing
    mps_time = np.array([float(el) for el in mps_time])
    mps_freqs = np.array([float(el) for el in mps_freqs])
    # axes labels
    ax1.title.set_text('Original MPS')
    ax2.title.set_text('Reconstructed MPS')
    # x labels
    x_inds = np.around(np.linspace(0, len(mps_time), 10, endpoint=False)).astype(int)
    ax1.set_xticks(x_inds)
    ax1.set_xticklabels(mps_time[x_inds])
    ax1.set_xlabel('modulation/s')
    ax2.set_xticks(x_inds)
    ax2.set_xticklabels(mps_time[x_inds])
    ax2.set_xlabel('modulation/s')
    # y labels
    y_inds = np.around(np.linspace(0, len(mps_freqs), 10, endpoint=False)).astype(int)
    ax1.set_yticks(y_inds)
    ax1.set_yticklabels(mps_freqs[y_inds])
    ax1.set_ylabel('modulation/s')
    ax2.set_yticks(y_inds)
    ax2.set_yticklabels(mps_freqs[y_inds])
    ax2.set_ylabel('modulation/Hz')
    # colorbar
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('log MPS')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('log MPS')
    # figure title
    if not fig_title is None:
        fig.suptitle(fig_title)
    return fig


def assess_predictions(orig_stim, predicted_stim, parameters, model_type):
    ''' Function computes user-defined measures of quality of predictions between 
    original and predicted data)
    Inputs:
        orig_stim - 2d numpy array of original stimulusrepresentation
        predicted_stim - 2d numpy array of predicted stimulus representation
        parameters - dictionary of stimulus parameters (output of the feature extractor)
    Outputs:
        assessments - dictionary of user specified assessment quantities
    '''
        
    # account for lagging in STM
    if model_type == 'STM':
        parameters["mps_shape"][0] = int(parameters["mps_shape"][0]/tmp_lag)
        parameters["mps_shape"][1] = int(parameters["mps_shape"][1]*tmp_lag)

    # compute correlation coefficient for every feature across different MPSs and plot it
    r2 = product_moment_corr(orig_stim, predicted_stim) 
    resh_r2 = lambda x: np.reshape(x, (parameters['mps_shape']))
    r2 = resh_r2(r2)
    fig_r2 = plot_score_across_features(r2, 'Correlation coefficient', parameters['mps_time'], parameters['mps_freqs']) 
     
    # compute correlation across different MPSs between reconstructed MPSs  and original MPSs and plot it
    correlations = compute_correlation(orig_stim, predicted_stim) 
    fig_cor = plot_score_across_mps(correlations, 'Correlation', [0,1])
    
    # denormalize original and reconstructed MPS
    denorm_orig_stim = denormalize(orig_stim, parameters)
    denorm_predicted_stim = denormalize(predicted_stim, parameters)
    
    # reshape original and reconstructed MPS
    orig_mps = reshape_mps(denorm_orig_stim, parameters['mps_shape'])
    reconstr_mps = reshape_mps(denorm_predicted_stim, parameters['mps_shape'])

    # plot denormalized original and reconstructed MPS with best, worst and "medium" correlation values 
    best_mps_ind = np.argmax(np.squeeze(np.abs(correlations)))
    best_mps = np.squeeze(reconstr_mps[:,:,best_mps_ind]) 
    fig_mps_best = plot_mps_and_reconstructed_mps(orig_mps[:,:,best_mps_ind], best_mps,\
        parameters['mps_time'], parameters['mps_freqs'], 'Best MPS')

    worst_mps_ind = np.argmin(np.squeeze(np.abs(correlations)))
    worst_mps = np.squeeze(reconstr_mps[:,:,worst_mps_ind]) 
     
    worst_mps[:,100] = worst_mps[:,100] - worst_mps[:,100] 
    fig_mps_worst = plot_mps_and_reconstructed_mps(orig_mps[:,:,worst_mps_ind], worst_mps,\
        parameters['mps_time'],  parameters['mps_freqs'], 'Worst MPS')

    medium_mps_ind = np.argmin(np.squeeze(np.abs(correlations-np.mean(correlations))))
    medium_mps = np.squeeze(reconstr_mps[:,:,medium_mps_ind])
    fig_mps_medium = plot_mps_and_reconstructed_mps(orig_mps[:,:,medium_mps_ind], medium_mps,\
        parameters['mps_time'], parameters['mps_freqs'], 'Medium MPS')
    
    figs = {"correlations": fig_cor, "R2": fig_r2, "Best_MPS": fig_mps_best,\
        "Worst_MPS": fig_mps_worst, "Medium_MPS": fig_mps_medium}  
    mps_inds = {'best_mps':int(best_mps_ind), 'worst_mps':int(worst_mps_ind), 'medium_mps':int(medium_mps_ind)}
    return {"correlations": correlations, "r2": r2, "figs": figs, "mps_inds": mps_inds}

def denormalize(stim, parameters):
    ''' Function adds mean and SD to the flattened stimulus representation by reading 
    it from stim_param file (output of feature extractor)
    Inputs:
        stim - 2d numpy array of flattened stimulus representation (time, MPSs)
        parameters - python dictionary (output of the feature extractor)
    Output:
        rescaled_stim - signal mutliplied by SD and incremented with mean of 
                          original signal
    '''
    mean = np.array(parameters["mps_mean"])
    sd = np.array(parameters["mps_sd"])
    if stim.shape[1] != mean.shape[0] or stim.shape[1] != sd.shape[0]:
        raise ValueError("The sizes of mean and sd arrays do not match the size\
            of the stimulus")
    stim = stim*sd + mean
    return stim


def params_are_equal(params_list):
    '''Function to check if all the parameters of feature extractor 
    in different runs of a signle subject are equal.
    Inputs:
        params - list of dictionaries (stim_param) - outputs of the feature 
                 extractor for each run
    Outputs:
        params_equal - logical, True if all the parameters are equal for every
                       dict in params list
    '''
    params_equal = True
    for run_num in range(0, len(params_list)-1):
        for key in params_list[run_num].keys():
            if not params_list[run_num][str(key)] == params_list[run_num+1][str(key)] and \
                key != 'mps_mean' and key != 'mps_sd':
                params_equal = False
                break
    return params_equal

def num_samples_are_equal(run_list):
    '''Function checks is number of fmri or stimulus representation samples (times)
    are equal between runs.
    Inputs:
        run_list - list of fmri or stimulus representations in different runs 
    Outputs:
        num_samples_equal - logical, true or false
    '''
    num_samples_equal = True
    for run_num in range(0, len(run_list)-1):
        if run_list[run_num].shape[0] != run_list[run_num+1].shape[0]:
            num_samples_equal = False
            break
    return num_samples_equal


### Custom decoder classes

class myRidge:
    def __init__(self, alphas = None, voxel_selection=True, var_explained=None, n_splits_gridsearch = 5, **kwargs):
        '''
        kwargs - additional arguments transferred to ridge_gridsearch_per_target
        alphas - list of 3 int (start, stop, num parameters in numpy logspace function),
                 optional. Default = 1000
                 Regularization parameters to be used for Ridge regression
        voxel_selection - bool, optional, default True
                          Whether to only use voxels with variance larger than zero.
        var_explained - float, whether to do pca on frmi with defiend variance explained. 
        Default = None (no pca)
        n_splits_gridsearch - int, number of cross-validation splits in ridge_gridsearch_per_target. '''
        self.alphas = alphas
        self.voxel_selection = voxel_selection
        self.n_splits_gridsearch = n_splits_gridsearch
        self.var_explained = var_explained
        self.pca = None
        self.ridge_model = None
        self.predicted_stim = None
    
    def fit(self, fmri, stim):
        '''Fit ridges to stimulus and frmi using k fold cross validation and voxel selection
        fmri - 2d numpy array of bold fmri data of shape (samples, voxels)
        stim - 2d numpy array of stimulus representation of shape (samples, features)
        '''
        if self.voxel_selection:
            voxel_var = np.var(fmri, axis=0)
            fmri = fmri[:,voxel_var > 0.]
        if self.alphas is None:
            self.alphas = [1000]
        else:
            self.alphas = np.logspace(self.alphas[0], self.alphas[1], self.alphas[2])
        if self.var_explained != None:
            # return pca object and keep it
            fmri, pca, _ = reduce_dimensionality(fmri, self.var_explained)
            self.pca = pca
        self.ridge_model = ridge_gridsearch_per_target(fmri, stim, self.alphas, n_splits = self.n_splits_gridsearch)
    
    def predict(self, fmri):
        ''' Returns the stimulus predicted from model trained on test set'''
        self.predicted_stim = self.ridge_model.predict(self.pca.transform(fmri))
        return self.predicted_stim


class myCCA(CCA):
    def __init__(self, n_components=10, **kwargs):
        super().__init__(n_components = n_components, **kwargs)
        self.predicted_data=None 
         
    def predict(self, fmri):
        self.predicted_data = CCA.predict(self, fmri)
        return self.predicted_data


### Decoding functions

def lag(stim, pred_sp, tmp_lag):
    '''Function introudces "laggin" by stacking tmp_lag rows of stim and spatial 
    predictions of stim horizontally.
    Inputs:
        stim - 2d numpy array of stimulus representation
        pred_sp - 2d numpy array of stimulus representation predicted from spatial
                  model
    Outputs:
        stim - 2d numpy array of "lagged" (reshaped) stimuli
        pred_sp - 2d numpy array of "lagged" (reshaped" stimulus predictions from
                  spatial model
    '''
    rem = pred_sp.shape[0] % tmp_lag
    if rem != 0:
        stim = stim[:rem,:]
        pred_sp = pred_sp[:rem,:]
        warnings.warn("tmp_lag does not fit the shape of the stimulus. The last "+\
            str(rem)+ " rows of stimulus and pred_sp were deleted.")
    stim = np.reshape(stim, (int(stim.shape[0]%tmp_lag), int(stim.shape[1]*tmp_lag)))
    pred_sp = np.reshape(pred_sp, (int(pred_sp.shape[0]%tmp_lag), int(pred_sp.shape[1]*tmp_lag)))
    return stim, pred_sp

def SM_decoding(fmri, stim, decoder, n_splits=8):
    '''Runs spatial model decoding on a single run of fmri and stimulus representation.
    Inputs:
        fmri - 2d numpy array of preprocessed fmri
        stim - 2d numpy array of  preprocessed stim
        decoder - CONFIGURED decoder object with .fit and .predict methods
                  to use in spatial model
        n_splits - int, number of cross-validation splits;
        Default = 8.
    Outputs:
        predictions - 2d numpy array of predicted stimulus representation
    '''
    # preallocate outputs
    decoders = []
    predictions = []
    # create cross-validation object
    kfold = KFold(n_splits=n_splits)
    for train, test in kfold.split(fmri, stim):
        # copy decoder object
        dec = copy.deepcopy(decoder)
        decoders.append(dec)
        # fit a copy of decoder object on train split
        decoders[-1].fit(fmri[train,:], stim[train,:])
        # predict stimulus from trained object
        predictions.append(decoders[-1].predict(fmri[test,:]))
        # concatenate predictions
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def STM_decoding(fmri, stim, decoder_sp, decoder_tmp, cv_sp, cv_tmp, tmp_lag):
    '''Function to run spatial - temporal model decoding on a single run of fmri and stimulus
    representation. The model first uses spatial decoder to predict stimulus representation 
    using spatial features. Then it "laggs" (stacks several rows into one) predicted stimulus
    representation and original stimulus representation and runs temporal decoder on it.
    Inputs:
        fmri - 2d numpy array of fMRI stimuli
        stim - 2d numpy array of stimulus representation
        decoder_sp - CONFIGURATED decoder object to use as spatial decoder
        decoder_tmp - CONFIGURATED decoder object to use as temporal decoder
        cv_sp - int or cv constructor object for spatial cross-validation 
        cv_tmp - int or cv constructor object for temporal cross-validation
        tmp_lag - int - lagging parameter for temporal model (~number of rows
                        of soatial model to stack into one in temporal model)
    Outputs:
        predictions_spatial - 2d numpy array of predictions from spatial model
        predictions_temporal - 2d numpy array of predictions from temporal model
    '''
    predictions_sp = []
    predictions_tmp = []
    cv_indices = {'train':[],'test':[]}
    # create cross-validation object
    kfold = KFold(n_splits=n_splits)
    for train, test in kfold.split(fmri, stim):
        
        # spatial decoder
        dec_sp = copy.deepcopy(decoder_sp)
        pred_sp = cross_val_predict(dec_sp, fmri, stim, cv=cv_sp)
        
        # "lagging" (reshaping)
        stim_lagged, pred_sp_lagged = lag(stim, pred_sp, tmp_lag)
        
        # temporal decoder
        dec_tmp = copy.deepcopy(decoder_tmp)
        pred_tmp = cross_val_predict(dec_tmp, pred_sp, stim, cv=cv_tmp)
        
        # append
        predictions_sp.append(pred_sp)
        predictions_tmp.append(pred_tmp)
        cv_indices["train"].append(train)
        cv_indices["test"].append(test)
         
    predictions_sp = np.concatenate(pred_sp, axis=0)
    predictions_tmp = np.concatenate(pred_tmp, axis=0)
    return predictions_sp, predictions_tmp


def stack_runs(fmri_runs, stim_runs, stim_param_list, subj_num):
    '''Function stacks fmri and stim from different runs into 2 2d numpy arrays
    and checks that stim parameters and numbers of samples are consistent between runs'''

    # make sure stim parameters are the same for all runs    
    if not params_are_equal(stim_param_list):
        raise ValueError("Stimuli parameters are not equal between runs for subject "+\
        str(subj_num) + "!")
    
    # make sure each run contains equal number of samples
    if not num_samples_are_equal(fmri_runs):
        raise ValueError("Number of samples of fmri data is not consistent between runs"+\
            " in subject " + str(subj_num) + "!")
    if not num_samples_are_equal(stim_runs):
        raise ValueError("Number of samples of stimulus represenation is not consistent"+\
        " between runs in subject " + str(subj_num) + "!")
    
    # concatenate multirun data
    fmri_multirun = np.concatenate(fmri_runs, axis=0)
    stim_multirun = np.concatenate(stim_runs, axis=0)
    return fmri_multirun, stim_multirun

def invoke_decoders(decoders, decoders_configs):
    '''Function creates decoder objects and configures them.
    Inputs:
        decoders - list of strings, decoder names
        decoders_configs - list of dictionaries - decoder-specific configs
    Outputs:
        invoked_decoders - list of configured decoder objects
    '''
    invoked_decoders = []
    for dec_num in range(len(decoders)):
        decoder = eval(decoders[dec_num])
        decoder = decoder(**decoders_configs[dec_num])
        invoked_decoders.append(decoder)
    return invoked_decoders

def decoding_model(inp_data_dir, out_dir, stim_param_dir, model_config, decoding_config):

    ### Input check (as model takes long time to run (would a be pity to find a glitch after running model for days:) )
    
    # unpack subjects and runs
    if "subjects"in model_config:
        subjects = model_config["subjects"]
    else:
        subjects =['01','02','03','04','05','06','09','14','15','16','17','18','19','20']
    
    if "runs" in model_config:
        runs = model_config["runs"]
    else:
        runs = [[1,2,3,4,5,6,7,8] for el in range(len(subjects))]
    
    if len(subjects) != len(runs):
        raise ValueError("Number of subjects does not match number of runs you have specified!"
        " Subjects shall be list of subject numbers and runs shall be list of lists of run numbers"
        "for every subject you have specified.")
    
    # loop through subject folders and runs and make sure all the files exist 
    for subj_counter, subj_num in enumerate(subjects):
        subj_folder = 'sub-'+ str(subj_num) 
        subj_folder_path = os.path.join(inp_data_dir, subj_folder)
        if not os.path.isdir(subj_folder_path):
            raise FileNotFoundError
        # Loop through runs of each subject
        for run_num in runs[subj_counter]:
            # get fmri, stimulus and stim_parameters paths and make sure they exist
            fmri = subj_folder + '_task-aomovie_run-' + str(run_num) + '_bold.tsv.gz'
            stim = 'task-aomovie_run-' + str(run_num) + '_stim.tsv.gz'
            stim_param = 'task-aomovie_run-' + str(run_num) + '_stim_parameters.json'
            fmri_path = os.path.join(subj_folder_path, fmri)
            stim_path = os.path.join(subj_folder_path, stim)
            stim_param_path = os.path.join(stim_param_dir, stim_param)
            if not os.path.isfile(stim_param_path):
                raise FileNotFoundError
            if not os.path.isfile(fmri_path):
                raise FileNotFoundError
            if not os.path.isfile(stim_path):
                raise FileNotFoundError
        # Check if subject folders in output dir shall be created
        if not os.path.isdir(os.path.join(out_dir, subj_folder)):
            print('Creating folder '+subj_folder+' in the directory '+out_dir)
            os.mkdir(os.path.join(out_dir, subj_folder))
        elif len(os.listdir(os.path.join(out_dir, subj_folder))) != 0:
            proceed = input('Directory ' + os.path.join(out_dir, subj_folder) + ' is not empty!\n Proceed? y/n \n')
            if proceed == 'n':
                return None
            elif proceed =='y':
                pass

    ### multirun_decoding
    fmri_multirun = []
    stim_multirun = []
    parameters_multirun=[]
    for subj_counter, subj_num in enumerate(subjects):
        subj_folder = 'sub-'+ str(subj_num) 
        subj_folder_path = os.path.join(inp_data_dir, subj_folder)
        # Loop through runs of each subject
        for run_num in runs[subj_counter]:
            # get fmri, stimulus and stim_parameters paths 
            fmri = subj_folder + '_task-aomovie_run-' + str(run_num) + '_bold.tsv.gz'
            stim = 'task-aomovie_run-' + str(run_num) + '_stim.tsv.gz'
            stim_param = 'task-aomovie_run-' + str(run_num) + '_stim_parameters.json'
            fmri_path = os.path.join(subj_folder_path, fmri)
            stim_path = os.path.join(subj_folder_path, stim)
            stim_param_path = os.path.join(stim_param_dir, stim_param)
            
            # Load data
            with open(fmri_path, 'rb') as fm:
                fmri = joblib.load(fm) 
            if not np.ndim(fmri) == 2:
                raise ValueError('Inputs shall be 2d numpy array' 
                'fmri shape was'+ str(X.shape)+ 'instead')
            with open(stim_path, 'rb') as st:
               stim = joblib.load(st)
            if not np.ndim(stim) == 2:
                raise ValueError('Inputs shall be 2d numpy array' 
                        'stimulus shape was'+ str(y.shape)+ 'instead')
            with open(stim_param_path, 'r') as par:
                parameters = json.load(par)

            # Append run
            fmri_multirun.append(fmri)
            stim_multirun.append(stim)
            parameters_multirun.append(parameters)
        
        # get concatenated multirun datasets
        fmri_multirun, stim_multirun = stack_runs(fmri_multirun, stim_multirun, parameters_multirun, subj_num)
        

        # check which model type to use
        if model_config["model_type"] == "SM": # spatial model
            print("running spatial model on subject "+ str(subjects[subj_counter]))
            decoders = invoke_decoders([decoding_config["decoder"]], [decoding_config["decoder_config"]])
            predictions = SM_decoding(fmri_multirun, stim_multirun, decoders[0], n_splits=decoding_config["n_splits"]) 
        
        elif model_config["model_type"] == "STM":
            print("running spatial-temporal model on subject "+str(subjects[subj_counter]))
            
            decoders = invoke_decoders([decoding_config["spatial_decoder"], decoding_config["temporal_decoder"]], \
                [decoding_config["spatial_decoder_config"], decoding_config["temporal_decoder_config"]])
            
            predictions_sp, predictions_tmpo = STM_decoding(fmri_multirun, stim_multirun, decoders[0],\
                decoders[1], decoding_config["cv_spatial"], decoding_config["cv_temporal"], decoding_config["temporal_lagging"])

            
        # assess predictions
        assessments = assess_predictions(stim, predictions, parameters)
        
        # save model data into subject-specific folder in the out_dir
        print('saving model data for subject ', str(subjects[subj_counter]) + "...")
        predictions_name = 'reconstructed_mps_run-' + str(run_num) + '.pkl'
        joblib.dump(predictions, os.path.join(out_dir, subj_folder, predictions_name))

        # save assessment data
        for assessment in assessments.keys():
            filename = assessment + '_run-' + str(run_num) + '.pkl'
            joblib.dump(assessments[assessment], os.path.join(out_dir, subj_folder, filename))

        # save figures (fig_cor, fig_mse, fig_mps_best, fig_mps_worst, fig_mps_medium) 
        for fig in assessments["figs"].keys():
            fig_name = fig + '_run-' + str(run_num) + '.png'
            fig_filename = os.path.join(out_dir, subj_folder, fig_name)
            assessments["figs"][fig].savefig(fig_filename, dpi = 300)
         

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Decoding model app. Reconstructs modulation power spectrum from aligned \n'
    'preprocessed stimulus and fmri data. User specifies input dir, output dir, model_config and decoding_config.\n'
    'model_config files shall be .json file containing subject list, run list and model type (SM for spatial and \n' 
    'STM for spatil-temporal. \n'
    'Note the default values for model_config parameters (if not specified in model_config file) are all subjects and runs. \n'
    'subjects 01, 02, 03, 04, 05, 06, 09, 10, 14, 15, 16, 17, 18, 19, 20 \n'  
    'runs 1, 2, 3, 4, 5, 6, 7, 8 \n'
    'decoding_config is model specific. See description below. \n'
    'IT IS RECOMMENDED TO STORE BOTH CONFIG FILES IN THE OUTPUT DIR.', formatter_class=argparse.RawTextHelpFormatter ) 

    parser.add_argument('-inp','--input_dir', type=str, help='Path to preprocessed stimuli and fmri')
    parser.add_argument('-out','--output_dir', type=str, help='Path to the output directory where the model \
    data shall be saved. Folder structure can be pre-created or absent')
    parser.add_argument('-stim_dir', '--stim_param_dir', type=str, help='Path to directory cantaining stimuli parameters. \n'
    'use the path where you feature extractor saved processed stimuli.')
    parser.add_argument('-mod_conf','--model_config', type=str, help='Path to json model config file containing \n'
    'stim_param_dir, subject list, list of lists of runs and decoder.') 
    parser.add_argument('-dec_conf', '--decoding_config', type=str, help='Path to the json file with decoding config')
    args = parser.parse_args()
    
    with open (args.decoding_config) as dconf:
        decoding_config = json.load(dconf)
    with open(args.model_config) as mconf:
        model_config = json.load(mconf)   
    
    # RUN DECODING MODEL WITH THE GIVEN ARGUMENTS   
    decoding_model(args.input_dir, args.output_dir, args.stim_param_dir, model_config, decoding_config)
