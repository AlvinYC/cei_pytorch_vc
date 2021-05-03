from pathlib import Path
from scipy.io import wavfile
from datetime import timedelta
from glob import glob
import numpy as np
import shutil

def get_duration_info(target_dir):
    p = Path(target_dir).glob('./*')
    files = [x for x in p if x.is_file()]
    sample_info = list(map(lambda x: wavfile.read(x), files)) #  sr, # of sample
    sec_info = list(map(lambda x: len(x[1])/x[0], sample_info))

    print(target_dir + ' have totally seconds: ' + str(timedelta(seconds=sum(sec_info))))
    return sample_info

def select_wav(target_dir,max_sec):
    p = Path(target_dir).glob('./*')
    files = [x for x in p if x.is_file()]
    sample_acc = 0
    select_files = []
    for wav_file in files:
        sr, sample = wavfile.read(wav_file)
        sample_acc += len(sample)
        if sample_acc > (max_sec*sr):
            return select_files
        else:
            select_files.append(wav_file)
    return True

def copy_sel(vctkdirs, cp_path, max_sec):
    for vctk in vctkdirs:
        sel_wavs = select_wav(vctk,max_sec)   # '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48/p225/'
        vctk_name = Path(vctk).name         # 'p225'
        Path.mkdir(Path(cp_path,vctk_name), parents=True, exist_ok=True) # '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_05min/wav48/p225'
        list(map(lambda x: shutil.copy(x, Path(cp_path,vctk_name)),sel_wavs))
    return True

     

if __name__== "__main__":
    # query duration for each vctk folder
    vctk_path = '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48'
    vctkdirs = sorted(glob(vctk_path+"/*/"))    # ['./p225/', './p226/', './p227/', './p376/']
    print(str(vctkdirs)+'\n')
    list(map(lambda x:get_duration_info(x), vctkdirs))

    # copy with limitation 
    cp_path = '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_05min/wav48'
    copy_sel(vctkdirs,cp_path,300)                      # ['/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48/p225/',...]

    # query duration for each new vctk folder    
    vctkdirs = sorted(glob(cp_path+'/*/')) 
    print(str(vctkdirs)+'\n')
    list(map(lambda x:get_duration_info(x), vctkdirs))
    


