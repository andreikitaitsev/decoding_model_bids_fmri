#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to distribute stimulus representation by folders according to the BIDS rules 
Script assumes already existing folder structure for your subjects.
'''
import os
import glob
import shutil
import sys
import argparse

__all__ = ['distribute_stimuli']

def distribute_stimuli(stimuli_dir, bids_dir):
    '''
    Non-interactive function to distribute stimuli from arbitrary directory by bids directory folders
    Inputs: 
    stimuli_dir - the directory of your bids project folder
    bids_dir    - path to the bids directory 
    '''
    # distribute tsv.gz files
    stimuli_files = glob.glob((stimuli_dir+"*.tsv.gz"))
    for file in stimuli_files:
        shutil.copy(file, bids_dir)
    
    # distribute stimuli description json files
    subj_folders = glob.glob((bids_dir + '/'+"*/")) #select all dirs
    subj_folders = [subj_folder for subj_folder in subj_folders if 'sub' in subj_folder]
    stimuli_description_files = glob.glob((stimuli_dir +'/'+'*description.json'))
    for subj_folder in subj_folders:
        subj_num = subj_folder[-7:-1] 
        for stim_file in stimuli_description_files:
            old_stim_name = stim_file.split('/')[-1]
            new_stim_name = subj_num +'_'+ old_stim_name
            shutil.copy(stim_file, subj_folder)
            os.rename(os.path.join(subj_folder,old_stim_name),\
            os.path.join(subj_folder, new_stim_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Distributer of precessed stimuli files by BIDS standard folders.')
    parser.add_argument('-s', '--stimuli_dir', help='directory where all the stimulus are saved by run_mps_feature_extraction.sh script',type=str)
    parser.add_argument('-b', '--bids_dir',help='path to the bids directory',type=str)
    args=parser.parse_args()
    stimuli_dir = args.stimuli_dir
    bids_dir = args.bids_dir
    distribute_stimuli(stimuli_dir, bids_dir)
