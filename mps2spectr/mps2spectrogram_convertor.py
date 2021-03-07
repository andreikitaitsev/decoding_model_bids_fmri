import numpy as np
import librosa
import soundfile
import joblib
import json
import os
import gzip
import matplotlib.pyplot as plt

def mps2spectr(mps, phases, **kwargs):
    '''
    Inputs:
        mps - 3d numpy array of shape (time, mps matrices), i.e
              (time, mps_freq, mps_time)
        phases - 3d numpy array of mps_phase information from feature extractor
        kwargs for ifft2 function
    Outputs:
        spectr_cum - 2d numpy array of spectrogram (freq, time)
    '''
    spectr_cum = []
    for time in range(mps.shape[0]):
        if mps[time].shape[0] % 2 != 0:
            mps_iter = np.vstack((np.flip(mps[time], 0)[1:,:], mps[time]))
        else:
            mps_iter = np.vstack((np.flip(mps[time], 0), mps[time]))
        # restore complex valued signal using phase information 
        mps_iter = mps_iter*(np.cos(phases[time]) + 1j*np.sin(phases[time]))
        spectr_iter = np.abs(np.fft.ifft2(np.fft.ifftshift(mps_iter),**kwargs))
        # account for negative freqs
        spectr_iter = spectr_iter[:mps_iter.shape[0]//2,:]
        spectr_iter[1:,:] = spectr_iter[1:,:]*2
        
        spectr_cum.append(spectr_iter)
    spectr_cum = np.concatenate(spectr_cum, axis = 1)
    return spectr_cum

def spectr2audio(spectr, hop_length_stft, **kwargs):
    '''
    Inputs: 
        spectr - 2d numpy array, spectrogram of shape (freqs,time)
        hop_length_stft - int, hop length used in feature extractor)
    Outputs:
        audio - 2d numpy array of audio time-series
    '''
    audio = librosa.istft(spectr, hop_length = hop_length_stft, **kwargs)
    return audio

def load_data(mps_path, stim_param_path, phase_path):
    mps=None
    params=None
    phases=None
    try:
        mps = joblib.load(mps_path)
    except:
        with gzip.open(mps_path, 'rt') as fl:
            mps = np.loadtxt(fl)
    try:
        with open(stim_param_path, 'r') as fl:
            params = json.load(fl)
    except:
        pass
    try:
        with gzip.open(phase_path, 'rt') as ph:
            phases = np.loadtxt(ph)
    except:
        pass
    return mps, params, phases

def mps2spectr_convertor(mps, params, phases):
    '''Converts mps into spectrogram.
    Inputs:
        mps - 2d numpy array of flattened MPS of shape (time, n_features) 
        params - dictionary of stim params
        phases - 2d numpy array of phases
    Outputs:
       spectrogram - 2d numpy array of spectrogram (freq, time)
    '''
    # denormalize
    mps = mps*np.broadcast_to(params["mps_sd"], mps.shape)
    mps = mps + np.broadcast_to(params["mps_mean"], mps.shape)
    if np.ndim(mps) != 3:
        mps = np.reshape(mps, (-1,params["mps_shape"][0],params["mps_shape"][1]))
    # reshape phases to 3d array (time, 2d pahse matrices) 
    phases = np.reshape(phases, (-1, params["mps_phase_shape"][0], params["mps_phase_shape"][1]))
    spectrogram = mps2spectr(mps, phases)
    return spectrogram 

def plot_orig_reconstr_spectr(orig_spectr, reconstr_spectr):
    '''PLots original and reconstructed spectrogram.'''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9))
    im1 = ax1.imshow(orig_spectr, origin='lower', aspect = 'auto')
    ax1.title.set_text('Original spectrogram')
    im2 = ax2.imshow(reconstr_spectr, origin='lower', aspect = 'auto')
    ax2.title.set_text('Reconstructed spectrogram')
    
    limits = [np.percentile(orig_spectr, 5), np.percentile(reconstr_spectr,95)]
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.mappable.set_clim(limits[0], limits[1])
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.mappable.set_clim(limits[0], limits[1])
    return fig



if __name__=='__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Convert original and reconstructed feature \n'
    'representation into spectrogram and plot it.')
    parser.add_argument('-orig','--orig_feature_dir',help='Directory with original feature representation.',\
        type=str)
    parser.add_argument('-reconstr','--reconstr_feature_path',type=str, help='PAth to file of reconstructed \n'
    'feature representation')
    parser.add_argument('-output','--output_dir',type=str, help='Directory where to save spectrogram pictures')
    args=parser.parse_args()
    
    # convert original feature representation into spectrogram
    orig_spectrogram = []
    orig_main_dir = '/data/akitaitsev/data1/raw_data/processed_stimuli/2'
    for runnum in range(1,9):
        mps_path = os.path.join(args.orig_feature_dir, ('task-aomovie_run-'+str(runnum)+'_stim.tsv.gz'))
        stim_param_path=os.path.join(args.orig_feature_dir,('task-aomovie_run-'+str(runnum)+'_stim_parameters.json'))
        mps_phase_path=os.path.join(args.orig_feature_dir, ('mps_phases_run-'+str(runnum)+'.tsv.gz'))
        mps, params, phases =  load_data(mps_path, stim_param_path, mps_phase_path)
        spectr = mps2spectr_convertor(mps, params, phases)
        orig_spectrogram.append(spectr)
    orig_spectrogram = np.concatenate(orig_spectrogram, axis=1)
    
    # convert reconstructed feature representation into spectrogram
    
    #stack mps phases into one array 
    phases=[]
    for runnum in range(1,9):
        with gzip.open(os.path.join(args.orig_feature_dir,('mps_phases_run-'+str(runnum)+'.tsv.gz')),'rt') as ph:
            phase = np.loadtxt(ph)
        phases.append(phase)
    phases=np.concatenate(phases, axis=0)
    params_path = os.path.join(args.orig_feature_dir,'task-aomovie_run-1_stim_parameters.json')
    reconstr_mps, params, _ = load_data(args.reconstr_feature_path, params_path, None)
    reconstr_spectrogram = mps2spectr_convertor(reconstr_mps, params, phases)
    
    # check if the output dir shall be created
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # small 30, 60 and 450 TR segments 
    fig1 = plot_orig_reconstr_spectr(orig_spectrogram[:, 31:60], reconstr_spectrogram[:, 31:60])
    fig1.savefig(os.path.join(args.output_dir,'reconstr_and_orig_spectr_SHORT.png'),dpi=300)
    fig2 = plot_orig_reconstr_spectr(orig_spectrogram[:, 31:120], reconstr_spectrogram[:, 31:120])
    fig2.savefig(os.path.join(args.output_dir,'reconstr_and_orig_spectr_MID.png'),dpi=300)
    fig3 = plot_orig_reconstr_spectr(orig_spectrogram[:, :450], reconstr_spectrogram[:, :450])
    fig3.savefig(os.path.join(args.output_dir,'reconstr_and_orig_spectr_LONG.png'),dpi=300)
    plt.show(block=True)
