#! usr/bin/env python3
# python script to copy bold files from aligned folder to data1/
import os
import shutil
import glob

def copy_files(from_dir, to_dir):
    for _, folders, __ in os.walk(from_dir): 
        for folder in folders: 
            if 'sub' in folder: 
                for file in os.listdir( os.path.join(from_dir,folder)):
                    if 'aomovie' in file and 'bold.nii.gz' in file:
                        shutil.copy(os.path.join(from_dir,folder,file),\
                        os.path.join(to_dir,folder))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='copy aomovoie bold files from aligned folder to your bids folder')
    parser.add_argument('-from','--from_dir', type=str)
    parser.add_argument('-to','--to_dir', type=str)
    args=parser.parse_args()
    from_dir = args.from_dir
    to_dir = args.to_dir
    copy_files(from_dir, to_dir)
