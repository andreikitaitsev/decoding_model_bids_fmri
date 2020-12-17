#! usr/bin/env python3
# Script to run preprocessing 
import joblib
import numpy as np
import os
import gzip
import glob
import json
import nibabel as nib
import sys
sys.path.append('/data/akitaitsev/data1/code/voxelwiseencoding/')
from preprocessing import preprocess_bold_fmri, make_X_Y
from nilearn.masking import compute_epi_mask

__all__=['run_preprocessing']

def run_preprocessing(input_dir, output_dir):
    ''' Function runs preprocessing by calling make_X_Y function for every bold 
    file and its correspondning stimuli file.
    Note, that if the output folder does not exist or does not have subfolder 
    structure (sub-01, etc.) it will be created.
    Inputs:
        input_dir -  path to directory, where the stimuli representation files and
                     bold files are stored
        output_dir - path to directory to store preprocessed stimuli representation 
                     and bold data
    '''
    # get BOLD TR
    bold_param_filename = os.path.join(input_dir,'task-aomovie_bold.json') 
    with open(bold_param_filename, 'r') as bfl:
        bold_params=json.load(bfl)
    TR=bold_params["RepetitionTime"]
    # loop through stimuli representation  files
    for _,__, stim_files in os.walk(input_dir):
        for stim_file in stim_files:
            # account for task-aomovie_bold.json
            if not '.json' in stim_file:
                stim_file_num = stim_file[stim_file.rfind("run-")+4]
                # loop through subject folders
                for _, subj_folders, __ in os.walk(input_dir):
                    for subj_folder in subj_folders:
                        if 'sub' in subj_folder:
                            # loop through bold files
                            for _,__, bold_files in os.walk(os.path.join(input_dir,subj_folder)):
                                for bold_file in bold_files:
                                    if 'bold.nii.gz' in bold_file:
                                        bold_file_num =bold_file[bold_file.rfind("run-")+4]
                                        # if stimulus file matches bold file
                                        if stim_file_num == bold_file_num:
                                            ## load data
                                            stim_param_filename=subj_folder+'_task-aomovie_run-'+ stim_file_num +'_stim_description.json'
                                            stim_param_file = os.path.join(input_dir,subj_folder,stim_param_filename)
                                            with open (stim_param_file,'r') as fl:
                                                stim_TR = json.load(fl)
                                                stim_TR = stim_TR["mps_repetition_time"]
                                            
                                            fmri = nib.load(os.path.join(input_dir, subj_folder, bold_file)) 
                                            
                                            with gzip.open(os.path.join(input_dir,stim_file), 'rt') as fl_stim:
                                                stimuli = np.loadtxt(fl_stim, delimiter='\t')   
                                            
                                            # compute mask - for now epi_mask for each run for each subject
                                            mask = compute_epi_mask(fmri)

                                            # preprocess BOLD
                                            fmri = preprocess_bold_fmri(fmri,mask=mask) 
          
                                            # make_X_Y without lagging
                                            stimuli, fmri = list([stimuli]), list( [fmri])
                                            lagged_stim, aligned_fmri = make_X_Y(stimuli, fmri, TR, stim_TR,lag_time=TR)
                                            
                                            # save lagged stimuli and aligned fmri to proper folder in the output directory
                                            lagged_bold_filename=subj_folder+'_task-aomovie_run-'+bold_file_num+'_bold.tsv.gz'
                                            with open(os.path.join(output_dir, subj_folder,lagged_bold_filename), 'wb') as fl_alfmri:
                                                joblib.dump(aligned_fmri, fl_alfmri)
                                            
                                            with open(os.path.join(output_dir, subj_folder,stim_file), 'wb') as fl_lagstim:
                                                joblib.dump(lagged_stim,fl_lagstim)
                                break                        
                    
                    break           
        break                              

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run preprocessing and lagging on all files in the input directory')
    parser.add_argument('-inp','--input_dir',type=str,help='input directory with bids data structure containing aligned frmi and feature representation')
    parser.add_argument('-out','--output_dir',type=str, help='output directory for preprocessed frmi andlagged (possibly) stimuli')
    args = parser.parse_args()
    run_preprocessing(args.input_dir, args.output_dir)


            
            
