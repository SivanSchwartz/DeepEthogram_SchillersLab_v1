import glob 
import cv2 
import os 
import pandas as pd 
import shutil
from itertools import chain
 
def CheckLengths(vids):
    
    total_frames, fps = [], [] 
    for vid in vids:
        v = cv2.VideoCapture(vid)
        total_frames.append(int(v.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps.append(int(v.get(cv2.CAP_PROP_FPS)))

    return total_frames, fps


def CheckCsvExist(vids):
    exist =[]
    for vid in vids:
        csv_name = vid.split('.')[0] + '_labels.csv'
        exist.append(os.path.isfile(csv_name))
        
    return exist

 
def copyFiles(paths, dst):
    for path in paths:
        # copy the csv
        csv_path = ''.join(glob.glob(path + '\*.csv'))
        vid_path = ''.join(glob.glob(path + '\*.mp4'))
        if csv_path != '':
            dest_path_csv = dst + '\\' + nameChange(csv_path, '.csv')
            shutil.copy(csv_path, dest_path_csv)
            # copy the vid 
            dest_path_vid = dst + '\\' + nameChange(csv_path, '.mp4')
            shutil.copy(vid_path, dest_path_vid)

        
def nameChange(filepath, filetype):
    # animal_exp_trail
    parts_name = ''.join(filepath).split('\\')
    animal_name = [name for name in parts_name if 'CT' in name]
    new_name = animal_name[0] + '_' + parts_name[-5] + '_' + parts_name[-2] + filetype
    new_name = '_'.join(new_name.split(' '))
    return new_name
    
    
    
def csvData2listtrails(csv_file):
    xls = pd.ExcelFile(csv_file)
    sheet_names = xls.sheet_names
    paths= []
    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet)
        for session in df.iloc[:,3]:
            # animal_name = session.split('\\')[-3] if 'CT' in session.split('\\')[-3] else session.split('\\')[-2]
            # session_project = '_'.join(session.split('\\')[-1].split(' ')) + '_deepethogram'
            # session_project = animal_name + '_' + session_project
            
            current_paths = glob.glob(session +'\\*deepethogram\\DATA\*')
            paths.append(current_paths)
    paths = list(chain(*paths))
    return paths



def main():
    csv_file = 'H:\\Models_deepEthogram\\add_data.xlsx'
    dst = 'H:\deepethogram\dataset'
    paths = csvData2listtrails(csv_file)
    copyFiles(paths, dst)
    
    vids = glob.glob(dst + '\*.mp4', recursive=True)
    total_frames, fps =  CheckLengths(vids)
    exist = CheckCsvExist(vids)

    data_dict = {'File path': vids, 
                'Total frames': total_frames,
                'FPS': fps,
                'CSV labels exist': exist}
    df = pd.DataFrame(data_dict)
    df.to_csv(dst + '\\DATACHECK.csv')
    
if __name__ == '__main__':
    main()

        
    