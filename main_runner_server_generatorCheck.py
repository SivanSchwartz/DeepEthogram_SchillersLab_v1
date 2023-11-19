import logging
import multiprocessing
import os
import random
import glob
import h5py
import shutil
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

def addDataTrain(project_config, vid_path, csv_path):
    list_of_movies = glob.glob(vid_path)
    csvs = glob.glob(csv_path)
    mode = 'copy' # or 'symlink' or 'move'
    # depending on the mode, it will copy, symlink, or move each video file
    # it will also compute the mean and standard deviation of each RGB channel
    for movie_path in list_of_movies:
        projects.add_video_to_project(project_config, movie_path, mode=mode)

    # change path to vids to be the new one 
    vids_project = project_config['project']['path'] + '/DATA/*/*.mp4'
    list_of_movies_updated = glob.glob(vids_project)

    for movie_path, label_path in zip(list_of_movies_updated, csvs):
        projects.add_label_to_project(label_path, movie_path)

def reset_logger():
  # First, overwrite any logger so that we can actually see log statements
  # https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
  log = logging.getLogger()  # root logger
  log.setLevel(logging.INFO)
  for hdlr in log.handlers[:]:  # remove all old handlers
      log.removeHandler(hdlr)
  log.addHandler(logging.StreamHandler())
  return log


def copyModels(project_config):
    try:
        src = '/home/sivan.s/DeepEthogramProject/deepethogram/MODELS_BACKBONE/pretrained_models' # change it to your path of backbone 
        dst = project_config['project']['path'] + '/models/pretrained_models'
        shutil.copytree(src, dst)
        print("Directory backbone pretrained models copied successfully!")
    except shutil.Error as e:
        print(f"Backbone pretrained model copy Error: {e}")
    except Exception as e:
        print(f"Unexpected error in copy backbone pretrained models: {e}")


def creatnewproject(data_path, project_name, behaviors):
    # this will create a folder called /mnt/DATA/open_field_deepethogram
    # there will be subdirectories called DATA and models
    # there will also be a project_config.yaml
    project_config = projects.initialize_project(data_path, project_name, behaviors)
    copyModels(project_config)
    return project_config

if __name__ == '__main__':
    project_path = r'/home/sivan.s/DeepEthogramProject/deepethogram/Models_deepEthogram/try100_deepethogram' # This is path example in my local machine 
    # # these are the real labels you have in the dataset 
    # behaviors = ['background', 
    #         'Perch', 
    #         'Lift', 
    #         'Reach', 
    #         'Grab_nonPellet',
    #         'Grab',
    #         'Sup',
    #         'AtMouth',
    #         'AtMouth_nonPellet',
    #         'BackPerch',
    #         'Table']

    # # give name to the project (folder will be created with this name + '_deepethogram')
    # project_name = 'Exp1'
    # project_config = creatnewproject(project_path, project_name, behaviors)

    # # Paths to the data you want to train on, just example here, change to your paths 
    # vid_path = r'/home/sivan.s/DeepEthogramProject/deepethogram/dataset/*.mp4'
    # csv_path = r'/home/sivan.s/DeepEthogramProject/deepethogram/dataset/*.csv'
    # addDataTrain(project_config, vid_path, csv_path)
    
    # stage 0: check dirs for project and initializations
    #project_path = project_config['project']['path'] # updated 
    files = os.listdir(project_path)
    assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
    assert 'models' in files, 'models directory not found! {}'.format(files)
    assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)
    
    log = reset_logger()
    print_dataset_info(os.path.join(project_path, 'DATA'))
        
    # check the gpus used 
    print(torch.__version__)
    print('gpu available: {}'.format(torch.cuda.is_available()))
    print('gpu name: {}'.format(torch.cuda.get_device_name(0)))
    assert torch.cuda.is_available(), 'Please select a GPU runtime and then restart!'

    # 1 stage: flow generator 
    preset = 'deg_f' # type of model -> deg_f, deg_m, deg_s
    cfg = configuration.make_flow_generator_train_cfg(project_path, preset=preset) 
    print(OmegaConf.to_yaml(cfg))
    n_cpus = multiprocessing.cpu_count()
    print('n cpus: {}'.format(n_cpus))
    cfg.compute.num_workers = n_cpus
    flow_generator = flow_generator_train(cfg)

