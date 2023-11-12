import logging
import multiprocessing
import os
import random
import glob
import h5py
import shutil
import argparse
# not used in DeepEthogram; only to easily show plots
#from IPython.display import Image
from omegaconf import OmegaConf
import pandas as pd
import torch
from deepethogram.sequence import train as trainseq
from deepethogram import configuration, postprocessing, projects, utils
from deepethogram.debug import print_dataset_info
from deepethogram.flow_generator.train import flow_generator_train
from deepethogram.feature_extractor.train import feature_extractor_train
from deepethogram.feature_extractor.inference import feature_extractor_inference
from deepethogram.sequence.train import sequence_train
from deepethogram.sequence.inference import sequence_inference


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # for new project initialization 
    parser.add_argument('--new_project', type=bool, default= True, help='create new project')
    parser.add_argument('--project_path', type=str, default= '', help='path to the new project where the config will be or is already located')
    parser.add_argument('--project_name', type=str, default= '', help='name for the new project')
    parser.add_argument('--vid_path', type=str, default= None, help='path to the videos')
    parser.add_argument('--csv_path', type=str, default= None, help='path to the csv labels')
    
    