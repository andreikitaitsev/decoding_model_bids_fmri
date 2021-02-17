import numpy as np
import librosa
import soundfile
import joblib
import json
import os
import gzip

def mps2spectr(mps, **kwargs):
    '''
    Inputs:
        mps - 3d numpy array of shape (time, mps matrices), i.e
              (time, mps_freq, mps_time)
        kwargs for ifft2 function
    Outputs:
        spectr_cum - 2d numpy array of spectrogram (freq, time)
    '''
    spectr_cum = []
    for time in range(mps.shape[0]):
        mps_iter = np.vstack((np.flip(mps[time], 0)[1:,:], mps[time]))
        spectr_iter = np.abs(np.fft.ifftshift(np.fft.ifft2(mps_iter,**kwargs)))
        spectr_iter = spectr_iter[spectr_iter.shape[0]//2 :, spectr_iter.shape[1]//2:] 
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

def mps2audio(mps_path, stim_param_path, filename, stim_type):
    '''Converts mps into audio and saves in under filename.
    Inputs:
        mps_path -str
        stim_param_path - str
        filename - str, output audio filename
        stim_type -str, orign or reconstr - scale by sd and add mean to 
                   original stimulus
    '''
    try:
        mps = joblib.load(mps_path)
    except:
        with gzip.open(mps_path, 'rt') as fl:
            mps = np.loadtxt(fl)
    with open(stim_param_path, 'r') as fl:
        params = json.load(fl)
    if stim_type == 'orig':
        mps = mps*np.broadcast_to(params["mps_sd"], mps.shape)
        mps = mps + np.broadcast_to(params["mps_mean"], mps.shape)
    if np.ndim(mps) != 3:
        mps = np.reshape(mps, (-1,params["mps_shape"][0],params["mps_shape"][1]))
    spectrogram = mps2spectr(mps)
    audio = spectr2audio(spectrogram, params["hop_length_stft"])
    soundfile.write(filename, audio, params["sr"])


if __name__=='__main__':
    raw_dir='/data/akitaitsev/data1/raw_data/processed_stimuli/'
    orig_mps_names = ['task-aomovie_run-'+str(el)+'_stim.tsv.gz' for el in range(1,9)]
    orig_mps_files = [os.path.join(raw_dir,str(el)) for el in orig_mps_names]
    orig_mps_params=['task-aomovie_run-'+str(el)+'_stim_parameters.json' for el in range(1,9)]
    orig_mps_params_files = [os.path.join(raw_dir, el) for el in orig_mps_params]
    out_dir = '/data/akitaitsev/data1/decoding_data5/audio/sub-01/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    wav_names = ['orig_'+str(el)+'.wav' for el in range(1,9)]
    out_dirs = [os.path.join(out_dir, el) for el in wav_names]
    for mps_path, stim_path, fname in zip(orig_mps_files, orig_mps_params_files, out_dirs):
        mps2audio(mps_path, stim_path, fname, 'orig')






















