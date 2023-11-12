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

class Train():
    def __init__(self, class_list, labeled_data_path, save_proj_path,
                 backboneWeightspath):
        self.vids_list = glob.glob(labeled_data_path + '\\**\\*.mp4',
                                   recursive=True)
        self.csv_list = glob.glob(labeled_data_path + '\\**\\*.csv',
                                   recursive=True)        
        self.class_list = class_list
        self.save_projects_path = save_proj_path
        self.backboneWeightspath = backboneWeightspath
    
    
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
    
    def CreatProjectName(self):
        
        # time getter 
        current_datetime = datetime.datetime.now()
        date_time_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        return 'Exp_'+date_time_string
    
    def creatnewproject(self,project_name):
    # this will create a folder called /mnt/DATA/open_field_deepethogram
    # there will be subdirectories called DATA and models
    # there will also be a project_config.yaml
        project_config = projects.initialize_project(self.save_projects_path,
                                                     project_name,
                                                     self.class_list)
        self.project_config = project_config
        self.CopyModelsBackbone()
        self.project_config = project_config
        return project_config
    
    def addDataTrain(self):
        list_of_movies =self.vids_list
        csvs = self.csv_list
        mode = 'copy' # or 'symlink' or 'move'
        # depending on the mode, it will copy, symlink, or move each video file
        # it will also compute the mean and standard deviation of each RGB channel
        for movie_path in list_of_movies:
            projects.add_video_to_project(self.project_config, movie_path, mode=mode)

        # change path to vids to be the new one 
        vids_project = self.project_config['project']['path'] + '/DATA/*/*.mp4'
        list_of_movies_updated = glob.glob(vids_project)

        for movie_path, label_path in zip(list_of_movies_updated, csvs):
            projects.add_label_to_project(label_path, movie_path)


    # NEW PROJECT CREATION, import this 
    def initialization(self):
        projectname = self.CreatProjectName()
        self.creatnewproject(projectname)
        self.addDataTrain()
        
        # stage 0: check dirs for project and initializations
        self.project_path = self.project_config['project']['path'] # updated 
        files = os.listdir(self.project_path)
        assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
        assert 'models' in files, 'models directory not found! {}'.format(files)
        assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)
        
        log = self.reset_logger()
        print_dataset_info(os.path.join(self.project_path, 'DATA'))
            
        # check the gpus used 
        print(torch.__version__)
        print('gpu available: {}'.format(torch.cuda.is_available()))
        print('gpu name: {}'.format(torch.cuda.get_device_name(0)))
        assert torch.cuda.is_available(), 'Please select a GPU runtime and then restart!'

    # TRAIN FLOW GENERATOR
    def train_flow_generation(self):
        preset = 'deg_f' # type of model -> deg_f, deg_m, deg_s
        cfg = configuration.make_flow_generator_train_cfg(self.project_path, preset = preset) 
                                                         
        print(OmegaConf.to_yaml(cfg))
        n_cpus = multiprocessing.cpu_count()
        print('n cpus: {}'.format(n_cpus))
        cfg.compute.num_workers = n_cpus # was n_cpus var
        flow_generator = flow_generator_train(cfg)
        
    def reset_logger(self):
        # First, overwrite any logger so that we can actually see log statements
        # https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
        log = logging.getLogger()  # root logger
        log.setLevel(logging.INFO)
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        log.addHandler(logging.StreamHandler())
        self.log = log
        return log
    
    
    # TRAIN FEATURE EXTRACTOR 
    def tain_feature_extractor(self):
        preset = 'deg_f' # type of model -> deg_f, deg_m, deg_s
        cfg = configuration.make_feature_extractor_train_cfg(self.project_path,
                                                             preset=preset)
        print(OmegaConf.to_yaml(cfg))
        # the latest string will find the most recent model by date
        # you can also pass a specific .pt or .ckpt file here
        cfg.flow_generator.weights = 'latest'
        n_cpus = multiprocessing.cpu_count()
        cfg.compute.num_workers = n_cpus

        log = self.reset_logger()
        feature_extractor = feature_extractor_train(cfg)


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

    # TRAIN SEQUENCE 
    def train_sequence(self):
        preset = 'deg_f'
        cfg = trainseq.make_sequence_train_cfg(self.project_path,
                                               use_command_line=True)
        trainseq.sequence_train(cfg)
 
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

    # POST PROCESS PLOTS
    def postprocessing(self):
        cfg = make_postprocessing_cfg(self.project_path)
        postprocess_and_save(cfg)
   

def main(class_list, 
        labeled_data_path,
        save_proj_path,
        backboneWeightspath):
    
    # create object 
    trainer = Train(class_list, 
                    labeled_data_path,
                    save_proj_path,
                    backboneWeightspath)
    
    trainer.initialization()
    trainer.train_flow_generation()
    trainer.tain_feature_extractor()
    trainer.inference_feature_extractor()
    trainer.train_sequence()
    trainer.inference_sequence()
    trainer.postprocessing()
    
    

if __name__ == '__main__':
    
    # inputs from user 
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
    
    labeled_data_path = r'C:\Users\Jackie.MEDICINE\Desktop\New folder'
    save_proj_path = r'H:\Models_deepEthogram' 
    backboneWeightspath = r'H:\Models_deepEthogram\MODELS_BACKBONE\pretrained_models'
    
    
    # create object 
    trainer = Train(class_list, 
                    labeled_data_path,
                    save_proj_path,
                    backboneWeightspath)
    
    trainer.initialization()
    trainer.train_flow_generation()
    trainer.tain_feature_extractor()
    trainer.inference_feature_extractor()
    trainer.train_sequence()
    trainer.inference_sequence()
    trainer.postprocessing()