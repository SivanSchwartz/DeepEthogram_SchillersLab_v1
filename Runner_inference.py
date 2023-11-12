import logging
import multiprocessing
import os
import random
import glob
import h5py
import shutil
import datetime
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
from deepethogram.postprocessing import postprocess_and_save
from deepethogram.configuration import make_postprocessing_cfg
from deepethogram.flow_generator.inference import make_feature_extractor_inference_cfg, flow_generator_inference
import argparse, yaml


class Inference():
    def __init__(self,class_list, new_data_path, proj_path,
                 backboneWeightspath):
        
        self.vids_list = glob.glob(new_data_path + '\\**\\trial*.mp4',
                                    recursive=True)
        self.csv_list = glob.glob(new_data_path + '\\**\\*.csv',
                                    recursive=True)        
        self.class_list = class_list
        self.project_path = proj_path
        self.backboneWeightspath = backboneWeightspath   
        
        with open(proj_path + '\project_config.yaml', 'r') as yaml_file:
            self.project_config = yaml.safe_load(yaml_file)     
    
    def CopyModelsBackbone(self):
        try:
            #src = 'H:\Models_deepEthogram\MODELS_BACKBONE\pretrained_models'
            src = self.backboneWeightspath
            dst = self.project_config['project']['path'] + '\models\pretrained_models'
            shutil.copytree(src, dst)
            print("Directory backbone pretrained models copied successfully!")
        except shutil.Error as e:
            print(f"Backbone pretrained model copy Error: {e}")
        except Exception as e:
            print(f"Unexpected error in copy backbone pretrained models: {e}")
    

    def addDataTrainInference(self):

        mode = 'copy' # or 'symlink' or 'move'
        # depending on the mode, it will copy, symlink, or move each video file
        # it will also compute the mean and standard deviation of each RGB channel
        
        # check if the trial already exist 
        train_vids = glob.glob(self.project_path + '\\DATA\\*\\*.mp4')
        train_vids = [os.path.basename(vid) for vid in train_vids]
        
        new_paths = [] 
        for movie_path in self.vids_list:
            if os.path.basename(movie_path) in train_vids:
                continue 
            new_path = projects.add_video_to_project(self.project_config, movie_path, mode=mode)    
            new_paths.append(new_path)
            
        for movie_path, label_path in zip(new_paths, self.csv_list):
            projects.add_label_to_project(label_path, movie_path)

    def reset_logger(self):
        # First, overwrite any logger so that we can actually see log statements
        # https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
        log = logging.getLogger()  # root logger
        log.setLevel(logging.INFO)
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        log.addHandler(logging.StreamHandler())
        return log


     # TRAIN FLOW GENERATOR
    def train_flow_generation(self):
        preset = 'deg_f' # type of model -> deg_f, deg_m, deg_s
        cfg = configuration.make_flow_generator_train_cfg(self.project_path,
                                                          preset=preset) 
        print(OmegaConf.to_yaml(cfg))
        n_cpus = multiprocessing.cpu_count()
        print('n cpus: {}'.format(n_cpus))
        cfg.compute.num_workers = n_cpus
        flow_generator = flow_generator_train(cfg)   
        
        
    # INFERENCE FEATURE EXTRACTOR 
    def inference_feature_extractor(self):
        preset = 'deg_f'
        cfg = configuration.make_feature_extractor_inference_cfg(project_path=self.project_path, preset=preset)
        print(OmegaConf.to_yaml(cfg))

        cfg.feature_extractor.weights = 'latest' 
        cfg.flow_generator.weights = 'latest' # do not change since it is adopted to the specific training 
        cfg.inference.overwrite = True
        # make sure errors are thrown
        cfg.inference.ignore_error = False
        cfg.compute.num_workers = 2
        feature_extractor_inference(cfg)

    # INFERENCE SEQUENCE MODEL
    def inference_sequence(self):
        preset = 'deg_f'
        cfg = configuration.make_sequence_inference_cfg(self.project_path)
        cfg.sequence.weights = 'latest'
        n_cpus = multiprocessing.cpu_count()
        cfg.compute.num_workers = n_cpus
        cfg.inference.overwrite = True
        cfg.inference.ignore_error = False
        sequence_inference(cfg)



    def inferenceTotal(self):
        self.addDataTrainInference()
        self.train_flow_generation()
        self.inference_feature_extractor()
        self.inference_sequence()
        # add to save results as csv 
        cfg = make_postprocessing_cfg(self.project_path)
        postprocess_and_save(cfg)

def main(class_list, new_data_path, proj_path,
                 backboneWeightspath):
    
    inference = Inference(class_list, new_data_path, proj_path,
                 backboneWeightspath)
    inference.inferenceTotal()
    
'''    
if __name__ == '__main__':
    
    class_list = ['background', 
            'Perch', 
            'Lift', 
            'Reach', 
            'Grab_nonPellet',
            'Grab',
            'Sup',
            'AtMouth',
            'AtMouth_nonPellet',
            'BackPerch',
            'Table']
    
    new_data_path = r'\\192.114.20.62\e\Maisan (Jackie-C-Analys)\2ph Experiments\Videos\CT93\fixed videos\23_04_14 HR tuft control\20230414_113845'
    proj_path = r'H:\Models_deepEthogram\Exp_2023_10_30_13_55_31_deepethogram'
    backboneWeightspath = r'H:\Models_deepEthogram\MODELS_BACKBONE\pretrained_models'
    inference = Inference(class_list, new_data_path, proj_path,
                 backboneWeightspath)
    
    inference.inferenceTotal()'''