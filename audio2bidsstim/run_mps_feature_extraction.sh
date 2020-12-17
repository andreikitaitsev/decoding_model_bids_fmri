#!/bin/bash

# change these parameters to your own configuration

workdir="/data/akitaitsev/data1/" # path to the root directory of the project
wav_path=$workdir"raw_data/stimuli/" # path to wav files
scriptpath=$workdir"code/audio2bidsstim/wav_files_to_bids_tsv_mps.py " #path to feature extractor script
wavfiles="0.wav 1.wav 2.wav 3.wav 4.wav 5.wav 6.wav 7.wav" # list of wav files to use      
outpath=$workdir"raw_data/processed_stimuli/" # tmp out path; later these files will be distributed according to bids folder structure 
# Read mps config - note in this case it is similar for every subject
# This config is adapted to have stimulus TR == fmri TR == 2s
mps_config=$workdir"code/audio2bidsstim/mps_input_config1.json"



# Run feature extraction - do not change this
if [ ! -d $workdir ]; then
    echo "Error: the directory "$workdir" does not exist!"
    return[]
fi

cd $workdir


# Run wav_files_to_bids_tsv_mps.py with this config for every wav file 
echo "Extracting features from audio files: "
    
for file in $wavfiles
do
    echo "extracting features from file "$file
    python $scriptpath $wav_path$file  -e "mps" -c $mps_config -o $outpath 
done
