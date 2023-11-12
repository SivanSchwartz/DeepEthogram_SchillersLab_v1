import glob 
import shutil
import os 

mother_dir = glob.glob(r'H:\Models_deepEthogram\Maisan_CT51_labeling_deepethogram\DATA\\*')

for dir in mother_dir:
    if 'MAIN' in dir:
        continue

    mother_name = dir.split('\\')[-1]
    vid_within = glob.glob(dir + '\\*\\*.mp4')
    csv_within = glob.glob(dir+ '\\*\\*.csv')

    # copy files 
    for vid,csv in zip(vid_within, csv_within):
        vid_dst = os.path.join(*vid.split('\\')[:-1]) + '\\' + mother_name +'-' + vid.split('\\')[-1]
        csv_dst = os.path.join(*csv.split('\\')[:-1]) + '\\' + mother_name +'-' + csv.split('\\')[-1]
        shutil.copy(vid, vid_dst)
        shutil.copy(csv, csv_dst)

    # change folders within 
    folder_within = glob.glob(dir + '\\*')
    os.rename(folder_within, mother_name+ '-' + folder_within)