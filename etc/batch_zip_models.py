# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 23:11:47 2026

@author: peter
"""
import os
import glob
import shutil

# Get all folders in the current directory
folders = [f for f in glob.glob("C:\\Users\\peter\\Desktop\\CascadeTorch\\CascadeTorch\\Pretrained_models\\*") if os.path.isdir(f)]
for folder in folders:
    folder_name = os.path.basename(os.path.normpath(folder))
    zip_filename = os.path.join(os.getcwd(), folder_name)
    # Create zip archive
    shutil.make_archive(zip_filename, 'zip', folder)
    print(f"Created: {zip_filename}.zip")



