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
import argparse, yaml


def addDataTrain(project_config, vid_path, csv_path):
    list_of_movies = glob.glob(vid_path)
    csvs = glob.glob(csv_path)
    mode = 'copy' # or 'symlink' or 'move'
    # depending on the mode, it will copy, symlink, or move each video file
    # it will also compute the mean and standard deviation of each RGB channel
    for movie_path in list_of_movies:
        projects.add_video_to_project(project_config, movie_path, mode=mode)

    # change path to vids to be the new one 
    vids_project = project_config['project']['path'] + '\DATA\*\*.mp4'
    list_of_movies_updated = glob.glob(vids_project)

    for movie_path, label_path in zip(list_of_movies_updated, csvs):
        projects.add_label_to_project(label_path, movie_path)

def addDataTrainInference(project_config, vid_path, csv_path):
    list_of_movies = glob.glob(vid_path)
    csvs = glob.glob(csv_path)
    mode = 'copy' # or 'symlink' or 'move'
    # depending on the mode, it will copy, symlink, or move each video file
    # it will also compute the mean and standard deviation of each RGB channel
    new_paths = [] 
    for movie_path in list_of_movies:
        new_path = projects.add_video_to_project(project_config, movie_path, mode=mode)
        new_paths.append(new_path)
    # change path to vids to be the new one 
    vids_project = project_config['project']['path'] + '\DATA\*\*.mp4'
    #list_of_movies_updated = glob.glob(vids_project)
    

    for movie_path, label_path in zip(new_paths, csvs):
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

# pre trained models 
def copyModels(project_config):
    try:
        src = 'H:\Models_deepEthogram\MODELS_BACKBONE\pretrained_models'
        dst = project_config['project']['path'] + '\models\pretrained_models'
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

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_data', type = str, default= '', help = 'The path to vids and labels')
    parser.add_argument('--vidType', type=str , default= '.mp4', help = 'Video format')
    parser.add_argument('--trainingProject', type=str , default= '', help = 'The training project to take weights from')


def main(debug = False):
    log = reset_logger()
       
    if not debug:
        args = args()
        # load all arguments 
        project_path = args.trainingProject
        project_config_path = args.trainingProject + '\project_config.yaml'
        with open(project_config_path, 'r') as yaml_file:
            project_config = yaml.safe_load(yaml_file)
        vid_path = args.new_data + '/**/*.' + args.vidType
        csv_path = args.new_data + '/**/*.csv' 
    
    else:
        # load all arguments 
        project_path = r'H:\Models_deepEthogram\CT99_23_06_09_HR_tuft_control_deepethogram'
        project_config_path = project_path + '\\project_config.yaml'
        with open(project_config_path, 'r') as yaml_file:
            project_config = yaml.safe_load(yaml_file)
        
        path_new_data = r'\\192.114.20.62\e\Maisan (Jackie-C-Analys)\2ph Experiments\Videos\CT99\fixed\23_05_23 HR tuft control\23_05_23_HR_tuft_control_deepethogram_deepethogram\DATA'
        vid_path = path_new_data + '/**/*.mp4' 
        csv_path = path_new_data + '/**/*.csv' 
    
    # add the inference data 
    addDataTrainInference(project_config, vid_path, csv_path)
        
    # check the gpus used 
    print(torch.__version__)
    print('gpu available: {}'.format(torch.cuda.is_available()))
    print('gpu name: {}'.format(torch.cuda.get_device_name(0)))
    assert torch.cuda.is_available(), 'Please select a GPU runtime and then restart!'

    # must trian the flow generator even if it is inferece 
    preset = 'deg_f' # type of model -> deg_f, deg_m, deg_s
    cfg = configuration.make_flow_generator_train_cfg(project_path, preset=preset) 
    print(OmegaConf.to_yaml(cfg))
    n_cpus = multiprocessing.cpu_count()
    print('n cpus: {}'.format(n_cpus))
    cfg.compute.num_workers = n_cpus
    flow_generator = flow_generator_train(cfg)


    # stage2 inference FEATURE EXTRACTOR 
    preset = 'deg_f'
    cfg = configuration.make_feature_extractor_inference_cfg(project_path=project_path, preset=preset)
    print(OmegaConf.to_yaml(cfg))

    cfg.feature_extractor.weights = 'latest' # TODO change to parser path -> I think it is OK sincce go to project last
    cfg.flow_generator.weights = 'latest' # do not change since it is adopted to the specific training 
    cfg.inference.overwrite = True
    # make sure errors are thrown
    cfg.inference.ignore_error = False
    cfg.compute.num_workers = 2
    feature_extractor_inference(cfg)
 
    # stage3 seq infierence 
    #project_path = r'H:\Models_deepEthogram\Test_runnerResults_deepethogram_withTheMissingLabels'
    preset = 'deg_f'
    cfg = configuration.make_sequence_inference_cfg(project_path)
    cfg.sequence.weights = 'latest'
    n_cpus = multiprocessing.cpu_count()
    cfg.compute.num_workers = n_cpus
    cfg.inference.overwrite = True
    cfg.inference.ignore_error = False

    sequence_inference(cfg)
    
    # load cfg and save all predictions 
    cfg = make_postprocessing_cfg(project_path)
    postprocess_and_save(cfg)



if __name__ == '__main__':
    main(debug=True)